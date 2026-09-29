"""考生信息区整体识别与解析

把一整个考生信息区（可能包含学校、班级、姓名、考场、准考证号、座号、条形码数字）
OCR 后的原始文本解析成结构化字段。
"""

import logging
import re
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

# 常见标签同义词
_NAME_LABELS = ["姓名", "名字", "考生姓名", "名"]
# 准考证号相关标签（输出到 exam_no）
_EXAM_NO_LABELS = ["准考证号", "准考证号码", "考号", "考生号"]
# 学号相关标签（输出到 student_no）
_STUDENT_NO_LABELS = ["学号", "报名号"]
_ROOM_LABELS = ["考场号", "考场", "考试场地", "试室", "考室"]
_SEAT_LABELS = ["座位号", "座号", "座位"]
_CLASS_LABELS = ["班级", "班级号"]
_SCHOOL_LABELS = ["学校", "中学", "初中", "高中"]

# 小语种/科目名称，无标签时不能当成姓名（含 OCR 常见误读：档/和/料）
_SUBJECT_WORDS = {"小语", "日语", "语文", "数学", "英语", "物理", "化学",
                  "生物", "政治", "历史", "地理", "外语", "语种", "小语种",
                  "小语档", "小语和", "小语料"}

# 小语种/科目字段标签（用于值截断与启发式清理，含 OCR 误读变体）
_MINOR_LANG_LABELS = ["小语种", "小语档", "小语和", "小语料", "语种", "科目"]

# 允许粘连取值的姓名标签（OCR 常丢失“姓名：”后的冒号，变成“姓名娜木汗”）
_GLUED_NAME_LABELS = ["考生姓名", "姓名", "名字"]

# OCR 引擎条码锚定标签条注入的显式标记（可信度最高的印刷体姓名，见 ocr.py recognize_block）
_MARKED_NAME_LABELS = ["印刷体姓名", "条码姓名"]

# OCR 引擎条码解码注入的显式考号标记（pyzbar 解码，比手写/印刷数字 OCR 可靠）
_BARCODE_NO_LABELS = ["条码考号"]


def parse_personal_info_block(raw_text: str,
                              extra_name_candidates: list = None,
                              candidate_names: list = None) -> Tuple[Dict[str, str], float]:
    """解析考生信息区 OCR 原始文本。

    Args:
        raw_text: 考生信息区 OCR 原始文本。
        extra_name_candidates: OCR 引擎提供的额外姓名候选（条码标签条多变体识别结果）。
        candidate_names: 该考试的考生名单（花名册），用于对姓名候选做相似度裁决。

    Returns:
        (fields, confidence)，fields 包含 name/student_no/exam_no/room/seat/class_name/school/raw_text。
    """
    logger.info("[parser-v2] 开始解析考生信息区, raw_text 长度=%s", len(raw_text) if raw_text else 0)
    if not raw_text:
        return {"raw_text": ""}, 0.0

    # 统一换行、去多余空格，保留换行用于分行
    text = raw_text.replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    flat = " ".join(lines)

    fields: Dict[str, str] = {"raw_text": raw_text}

    # 1. 座号：座号/座位号 后接 1-3 位数字（前后不能再是数字，避免吞掉考号 802200211）
    seat = _extract_by_label(flat, _SEAT_LABELS, r"(?<!\d)\d{1,3}(?!\d)")
    if seat:
        fields["seat"] = seat

    # 2. 考场：只取标签后的连续数字，避免把粘连的“小语和：日语”吸进来
    room = _extract_by_label(flat, _ROOM_LABELS, r"(?<!\d)\d{1,4}(?!\d)")
    if not room:
        room_match = re.search(r"第\s*([一二三四五六七八九十0-9]+)\s*考场", flat)
        if room_match:
            room = _cn_to_arabic(room_match.group(1))
    if room:
        fields["room"] = room

    # 3. 准考证号/学号：
    #    - 先按标签提取，区分准考证号（exam_no）和学号（student_no）；
    #    - 无标签时取全文最长数字串，考试场景默认当作准考证号；
    #    - 两者取最长，避免 "准考证号22021" 这种粘连误读覆盖真正的条码 320210011。
    nums = re.findall(r"\d{6,20}", flat)
    longest_num = max(nums, key=len) if nums else ""

    label_exam_no = _extract_by_label(flat, _EXAM_NO_LABELS, r"[A-Za-z0-9\-]{4,20}")
    label_student_no = _extract_by_label(flat, _STUDENT_NO_LABELS, r"[A-Za-z0-9\-]{4,20}")
    # 条码解码值（pyzbar）优先级最高，避免手写/印刷数字 OCR 误识（80220G229 等）
    barcode_exam_no = _extract_by_label(flat, _BARCODE_NO_LABELS, r"[A-Za-z0-9\-]{4,20}")

    exam_no = barcode_exam_no or label_exam_no or ""
    student_no = label_student_no or ""

    # 如果最长数字比标签提取的更长，用它补充缺失的字段（条码命中时不再被覆盖）
    if longest_num and not barcode_exam_no:
        if len(longest_num) > len(exam_no):
            exam_no = longest_num
        if len(longest_num) > len(student_no):
            student_no = longest_num

    # 只有标签明确为学号时，才把最长数字同时作为学号；否则默认仅保留 exam_no
    if not label_student_no:
        student_no = ""

    if exam_no:
        fields["exam_no"] = exam_no
    if student_no:
        fields["student_no"] = student_no

    # 4. 班级：优先按“班级”标签提取，其次匹配 X班/几年级X班
    class_name = _extract_by_label(flat, _CLASS_LABELS, r"[^\s：:]+(?:\s+[^\s：:]+)?")
    if not class_name:
        class_match = re.search(r"([一二三四五六七八九十\d]+(?:年级|年)?[（(]?[一二三四五六七八九十\d]+[）)]?\s*班)", flat)
        if class_match:
            class_name = class_match.group(1).strip()
    if not class_name:
        simple_class_match = re.search(r"(\d+)\s*班", flat)
        if simple_class_match:
            class_name = simple_class_match.group(1) + "班"
    if class_name:
        fields["class_name"] = class_name

    # 5. 学校：优先按“学校”标签提取；其次从考场标签前面提取；最后全文兜底。
    #    若提取结果包含“考场”或与考场号相同，则丢弃，避免把考场当学校。
    room_value = fields.get("room", "")
    school = _extract_by_label(flat, _SCHOOL_LABELS, r"[^\s：:]+(?:\s+[^\s：:]+)?")
    if not school:
        school = _extract_school_by_room_label(flat, room_value)
    if not school:
        school_match = re.search(
            r"((?:[^\s：:]{2,8}(?:中学|学校|初中|高中))|(?:[^\s：:]{1,8}附中)|"
            r"(?:[0-9一二三四五六七八九十]{1,8}中)|(?:[^\s：:]{2,8}中))(?:\s|$)",
            flat,
        )
        if school_match:
            school = school_match.group(1).strip()
    if school and ("考场" in school or school == room_value):
        school = ""
    if school:
        fields["school"] = school

    # 6. 姓名：
    #    条码锚定标签条注入的显式标记（印刷体，最可靠）> 打印条码标签行（印刷体）
    #    > “姓名：XXX”手写标签 > 粘连标签 > 启发式兜底。
    name = _extract_by_label(flat, _MARKED_NAME_LABELS, r"[\u4e00-\u9fa5·]{2,8}")
    if not name:
        name = _extract_name_by_printed_label(flat)
    if not name:
        # 优先匹配“姓名：XXX”标签
        name = _extract_by_label(flat, _NAME_LABELS, r"[^\s：:]+(?:\s+[^\s：:]+)?")
    if not name:
        # OCR 丢失冒号/空格分隔时（“姓名娜木汗”），标签后的负向预查会挡住中文名，
        # 需要单独允许姓名标签后直接粘连中文取值
        name = _extract_name_by_glued_label(flat)
    if not name:
        # 没有标签时，把已识别的字段从文本中去掉，剩下的中文里取最可能是姓名的 2-8 字词
        name = _extract_name_heuristic(flat, fields)
    if name:
        fields["name"] = name

    # 7. 花名册裁决：把所有姓名候选（标签条多变体 + 解析链结果）对考生名单做相似度匹配，
    #    取最优者。OCR 对少数民族低频人名天然不稳（敖其泰→放其泰），但候选里通常有一个
    #    足够接近真名；名单匹配可把单字符误识全部纠正回来。
    if candidate_names:
        pool = []
        for n in list(extra_name_candidates or []) + [name]:
            if n and n not in pool:
                pool.append(n)
        matched = _best_roster_match(pool, candidate_names)
        if matched:
            if matched != name:
                logger.info("[parser] 花名册裁决姓名: %s -> %s (候选=%s)", name, matched, pool)
            fields["name"] = matched

    # 置信度：解析出的字段越多置信度越高（exam_no/student_no 任一命中即可）
    core_fields = [
        fields.get("name"),
        fields.get("exam_no") or fields.get("student_no"),
        fields.get("room"),
        fields.get("seat"),
    ]
    confidence = sum(1.0 for v in core_fields if v) / len(core_fields)

    return fields, round(confidence, 4)


def _label_pattern(label: str, value_pattern: str) -> str:
    """构造标签匹配正则。

    只限制标签后不能紧跟中文字符（避免“考场”匹配到“考场号”前缀）；
    标签前允许中文，以便处理“二连三中考场号：1”这种学校名粘连在标签前的情况。
    """
    return rf"{re.escape(label)}(?![\u4e00-\u9fa5])[：:\s]*({value_pattern})"


def _extract_by_label(text: str, labels: list, value_pattern: str = r"[^\s：:]+(?:\s+[^\s：:]+)?") -> str:
    """按“标签[:：]值”模式提取值。"""
    for label in labels:
        pattern = _label_pattern(label, value_pattern)
        m = re.search(pattern, text)
        if m:
            val = m.group(1).strip()
            # 去掉尾部常见标签/符号
            val = re.sub(r"[：:]$", "", val)
            # 若值里出现了其它字段的标签开头，则截断（避免 OCR 行连粘）
            val = _truncate_at_next_label(val, labels)
            return val
    return ""


def _truncate_at_next_label(value: str, current_labels: list) -> str:
    """如果值中出现了其它字段的标签开头，截断到该标签之前。"""
    all_labels = set(_NAME_LABELS + _STUDENT_NO_LABELS + _ROOM_LABELS + _SEAT_LABELS + _CLASS_LABELS + _SCHOOL_LABELS + _MINOR_LANG_LABELS)
    other_labels = all_labels - set(current_labels)
    for label in sorted(other_labels, key=len, reverse=True):
        idx = value.find(label)
        if idx > 0:
            return value[:idx].strip()
    return value


def _extract_name_by_glued_label(text: str) -> str:
    """处理 OCR 丢失分隔符的“姓名娜木汗”场景：姓名标签后直接粘连中文名。

    只保留取值开头的中文部分，并截断到下一个字段标签（座位号/小语种等）之前，
    避免把粘连的后续字段吸进姓名。
    """
    for label in _GLUED_NAME_LABELS:
        m = re.search(rf"{re.escape(label)}[：:\s]*([^\s：:]+)", text)
        if not m:
            continue
        val = m.group(1).strip()
        val = _truncate_at_next_label(val, _NAME_LABELS)
        # 姓名只可能是中文（含间隔号），去掉粘连的数字/字母/标点
        m2 = re.match(r"^[\u4e00-\u9fa5·]+", val)
        if m2 and len(m2.group(0)) >= 2:
            return m2.group(0)
    return ""


# 印刷标签行姓名锚点：优先“班级:”，其次“考场:”“座位:”（OCR 可能漏识别“班级”字样）
_PRINTED_LABEL_ANCHORS = ["班级", "考场", "座位"]

# 印刷标签锚点前可能出现的字段词/科目词，不能当作姓名
_PRINTED_LABEL_STOP_WORDS = {"班级", "考场", "座位", "座号", "姓名", "学校", "年级"}


def _name_similarity(a: str, b: str) -> float:
    """姓名相似度：同长度且仅差 1 字（OCR 单字误识的典型形态，如 放其泰/敖其泰）
    视为完全可信；否则用编辑比率。"""
    import difflib

    if len(a) == len(b) and len(a) >= 3:
        diff = sum(1 for x, y in zip(a, b) if x != y)
        if diff <= 1:
            return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _best_roster_match(pool: list, roster: list, cutoff: float = 0.75) -> str:
    """从候选池中选出与花名册最相似的名字。

    逐候选打分（与名单全体成员取最大相似度），返回得分最高的名单名。
    阈值设计（防止名单中没有真名时误配到别人，如 查干其其格→苏都其其格）：
    - 两字名要求 ratio ≥ 0.75；
    - 三字及以上默认 ratio ≥ 0.75，但同长度仅差 1 字直接视为命中（OCR 单字误识）。
    无候选过阈值时返回空串（保留原 OCR 结果）。
    """
    best_name, best_score = "", 0.0
    for cand in pool:
        if not cand:
            continue
        for ref in roster:
            if not ref:
                continue
            if cand == ref:
                return ref  # 完全一致直接命中
            score = _name_similarity(cand, ref)
            if score >= cutoff and score > best_score:
                best_name, best_score = ref, score
    return best_name


def best_roster_match(pool: list, roster: list, cutoff: float = 0.6) -> str:
    """公开包装：从候选池中选出与花名册最相似的名字（供 OCR 引擎/服务层复用）。"""
    return _best_roster_match(pool, roster, cutoff)


def upsert_marked_name(raw_text: str, name: str) -> str:
    """把最终姓名以“印刷体姓名”标记行回写 raw_text（移除旧标记），
    保证 Java 端对 student_info_block 原文的二次解析与 Python 端结论一致。"""
    lines = [l for l in (raw_text or "").split("\n")
             if not re.match(r"^\s*(印刷体姓名|条码姓名)[:：]", l)]
    lines.append(f"印刷体姓名:{name}")
    return "\n".join(lines)


def extract_printed_label_name(text: str) -> str:
    """公开包装：从印刷标签行提取姓名（供 OCR 引擎判断是否需要补充识别）。"""
    return _extract_name_by_printed_label(text)


def _extract_name_by_printed_label(text: str) -> str:
    """从打印条码标签行提取姓名（打印体，识别稳定，优先于手写行 OCR 结果）。

    典型格式：“{学校/前缀} {姓名} 班级:{班级}{考号} 考场:{N}班 座位:{M}”，如：
      “韩-中 查干其其格 班级:九班802200235 考场:341班 座位:25”
      “东转一中 嘎尔帝 班级:八班802200237 考场:341班 座位:27”
    姓名位于锚点标签前的末段（前面的段是学校名/民族-语种前缀，段数不固定）。
    “班级:”锚点缺失时（OCR 漏识别），依次尝试“考场:”“座位:”锚点。
    """
    for anchor in _PRINTED_LABEL_ANCHORS:
        name = _extract_name_before_anchor(text, anchor)
        if name:
            return name
    return ""


def _extract_name_before_anchor(text: str, anchor: str) -> str:
    """取指定锚点标签前的最后一个中文分段作为候选姓名，并做合法性校验。"""
    m = re.search(rf"([\u4e00-\u9fa5·\-\s]*[\u4e00-\u9fa5·])\s*{anchor}[:：]", text)
    if not m:
        return ""
    seg = m.group(1).strip()
    parts = [p for p in re.split(r"\s+", seg) if p]
    if not parts:
        return ""
    name = parts[-1]
    # 单段内残留“韩-中 ”这类 X-X 前缀时剥掉
    name = re.sub(r"^[\u4e00-\u9fa5]{1,4}-[\u4e00-\u9fa5]{1,4}\s+", "", name)
    # 姓名只可能是中文（含间隔号），2-8 字
    if not re.fullmatch(r"[\u4e00-\u9fa5·]{2,8}", name):
        return ""
    # 锚点前的字段词/科目词不能当作姓名
    if name in _PRINTED_LABEL_STOP_WORDS or name in _SUBJECT_WORDS:
        return ""
    # 姓名段 OCR 丢失时末段会落到学校名/班级词（如“东转一中”“九班”），此时拒绝取值
    if name.endswith(("中学", "学校", "初中", "高中", "附中", "班")) or _looks_like_school_abbr(name):
        return ""
    return name


def _extract_name_heuristic(text: str, fields: Dict[str, str]) -> str:
    """无标签时，用启发式提取姓名。"""
    # 去掉已识别字段的文本，减少干扰
    removed = text
    for key in ("student_no", "room", "seat", "class_name", "school"):
        val = fields.get(key)
        if val:
            removed = removed.replace(val, " ")

    # 去掉纯数字串和常见标签（含小语种/科目字段，避免 OCR 误读成姓名）
    removed = re.sub(r"\b\d+\b", " ", removed)
    for label in _NAME_LABELS + _STUDENT_NO_LABELS + _ROOM_LABELS + _SEAT_LABELS + _CLASS_LABELS + _SCHOOL_LABELS + _MINOR_LANG_LABELS:
        removed = removed.replace(label, " ")

    # 按空白切分，找出纯中文（含间隔号）2-8 字词（蒙古族姓名常见 5 字以上，如“查干其其格”）
    candidates = []
    for token in re.split(r"[\s：:,，]+", removed):
        token = token.strip()
        if token and 2 <= len(token) <= 8 and re.fullmatch(r"[\u4e00-\u9fa5·]+", token):
            candidates.append(token)

    if not candidates:
        return ""

    # 常见学校/班级/科目后缀或整词，排除
    exclude_suffixes = ("中学", "学校", "初中", "高中", "班级", "班", "考场", "座号")
    for c in candidates:
        if c in _SUBJECT_WORDS:
            continue
        # 形如“连三中”“市一中”“附中”等学校简称，不应作为姓名
        if _looks_like_school_abbr(c):
            continue
        if not any(c.endswith(s) for s in exclude_suffixes):
            return c
    # 所有候选都不符合姓名特征，说明文本中可能没有姓名
    return ""


def _looks_like_school_abbr(token: str) -> bool:
    """判断 token 是否是学校简称（X中/附中/市一中 等）。"""
    return bool(re.fullmatch(
        r"(?:[0-9一二三四五六七八九十]+中)|(?:[^一-龥\s：:]*[一-龥]{1,6}中)|附中",
        token,
    ))


def _extract_school_by_room_label(text: str, room_value: str) -> str:
    """根据“考场/考场号”标签位置，提取其前面的学校名。

    常见格式：
      二连三中考场号：1 ...
      实验中学 考场：3 ...
    """
    if not text:
        return ""
    # 找到第一个考场标签的位置
    first_idx = -1
    matched_label = ""
    for label in _ROOM_LABELS:
        pattern = _label_pattern(label, r"")
        m = re.search(pattern, text)
        if m:
            idx = m.start()
            if first_idx == -1 or idx < first_idx:
                first_idx = idx
                matched_label = label
    if first_idx <= 0:
        return ""

    prefix = text[:first_idx].strip()
    # 从 prefix 末尾提取学校名：支持“实验中学”“第二中学”“连三中”“附中”等
    school_match = re.search(
        r"((?:[^\s：:]{2,8}(?:中学|学校|初中|高中))|(?:[^\s：:]{1,8}附中)|"
        r"(?:[0-9一二三四五六七八九十]{1,8}中)|(?:[^\s：:]{2,8}中))\s*$",
        prefix,
    )
    if school_match:
        return school_match.group(1).strip()
    return ""


def _cn_to_arabic(cn: str) -> str:
    """简单中文数字转阿拉伯数字，失败则原样返回。"""
    mapping = {
        "一": "1", "二": "2", "三": "3", "四": "4", "五": "5",
        "六": "6", "七": "7", "八": "8", "九": "9", "十": "10",
        "0": "0", "1": "1", "2": "2", "3": "3", "4": "4",
        "5": "5", "6": "6", "7": "7", "8": "8", "9": "9",
    }
    out = ""
    for ch in cn:
        out += mapping.get(ch, ch)
    digits = re.sub(r"[^0-9]", "", out)
    return digits if digits else cn
