"""考生信息区解析器单元测试。"""

import unittest

from omr_service.engine.personal_info_block_parser import parse_personal_info_block


class TestPersonalInfoBlockParser(unittest.TestCase):
    """覆盖用户反馈的典型 OCR 粘连/学校简称场景。"""

    def test_school_abbr_without_number_prefix(self):
        """学校简称首字被 OCR 漏掉时仍应识别为学校，不污染姓名。"""
        raw = "连三中考场号：2小语：日语 姓名：佟子怡 座位号：54 320220054"
        fields, _ = parse_personal_info_block(raw)
        self.assertEqual("佟子怡", fields.get("name"))
        self.assertEqual("连三中", fields.get("school"))
        self.assertEqual("2", fields.get("room"))
        self.assertEqual("54", fields.get("seat"))
        self.assertEqual("320220054", fields.get("exam_no"))

    def test_school_abbr_with_number_prefix(self):
        """完整学校名（中文数字+简称）可正常识别。"""
        raw = "二连三中考场号：2小语：日语 姓名：佟子怡 座位号：54 320220054"
        fields, _ = parse_personal_info_block(raw)
        self.assertEqual("佟子怡", fields.get("name"))
        self.assertEqual("二连三中", fields.get("school"))

    def test_user_case_namuhan(self):
        """用户截图中娜木汗的实际识别文本：学校简称缺少首字。"""
        raw = "连三中考场号：2小语：日语 姓名：娜木汗 座位号：65 320220065 320220065"
        fields, _ = parse_personal_info_block(raw)
        self.assertEqual("娜木汗", fields.get("name"))
        self.assertEqual("连三中", fields.get("school"))
        self.assertEqual("65", fields.get("seat"))

    def test_full_school_name(self):
        """常规“XX中学”学校名仍可识别。"""
        raw = "实验中学考场号：1 姓名：张三 座位号：5 320210005"
        fields, _ = parse_personal_info_block(raw)
        self.assertEqual("张三", fields.get("name"))
        self.assertEqual("实验中学", fields.get("school"))

    def test_school_with_cn_number(self):
        """“第二中学”类学校名仍可识别。"""
        raw = "第二中学考场号：1 姓名：李四 座位号：10 320210010"
        fields, _ = parse_personal_info_block(raw)
        self.assertEqual("李四", fields.get("name"))
        self.assertEqual("第二中学", fields.get("school"))

    def test_school_fuzhong(self):
        """“附中”类简称可识别。"""
        raw = "师大附中 考场号：3 姓名：王五 座位号：1 320210031"
        fields, _ = parse_personal_info_block(raw)
        self.assertEqual("王五", fields.get("name"))
        self.assertEqual("师大附中", fields.get("school"))

    def test_subject_words_not_name(self):
        """无姓名标签时，不能把“日语/小语”等科目名当作姓名。"""
        raw = "连三中考场号：2小语：日语 座位号：54 320220054"
        fields, _ = parse_personal_info_block(raw)
        self.assertEqual("连三中", fields.get("school"))
        # 姓名缺失，不能错认成“日语”或“小语”
        self.assertNotIn(fields.get("name"), {"日语", "小语", "小语种"})

    def test_name_glued_to_label_and_subject_misread(self):
        """用户截图实际文本：冒号丢失“姓名娜木汗” + “小语种”被 OCR 误读为“小语档”。"""
        raw = "二连三中考场号： 2小语档： 日语 姓名娜木汗 座位号： 65 320220065"
        fields, _ = parse_personal_info_block(raw)
        self.assertEqual("娜木汗", fields.get("name"))
        self.assertEqual("二连三中", fields.get("school"))
        self.assertEqual("2", fields.get("room"))
        self.assertEqual("65", fields.get("seat"))
        self.assertEqual("320220065", fields.get("exam_no"))

    def test_name_glued_with_following_label(self):
        """姓名与后续字段完全无空格时，应在下一个标签处截断。"""
        raw = "二连三中 考场号：2 姓名娜木汗座位号：65 320220065"
        fields, _ = parse_personal_info_block(raw)
        self.assertEqual("娜木汗", fields.get("name"))
        self.assertEqual("65", fields.get("seat"))

    def test_subject_misread_variant_not_name(self):
        """无姓名标签时，“小语种”的 OCR 误读变体也不能当作姓名。"""
        raw = "二连三中考场号：2小语档：日语 座位号：65 320220065"
        fields, _ = parse_personal_info_block(raw)
        self.assertNotIn(fields.get("name"), {"日语", "小语档", "小语和", "小语料"})

    def test_name_from_printed_barcode_label_long_name(self):
        """手写姓名行 OCR 丢失时，应从打印条码标签行提取姓名（用户实况：5 字蒙古名查干其其格）。"""
        raw = "韩-中 查干其其格 班级:九班802200235 考场:341班 座位:25\n802200235"
        fields, _ = parse_personal_info_block(raw)
        self.assertEqual("查干其其格", fields.get("name"))
        self.assertEqual("25", fields.get("seat"))
        self.assertEqual("341", fields.get("room"))
        self.assertEqual("802200235", fields.get("exam_no"))

    def test_name_from_printed_label_without_prefix(self):
        """打印条码标签无民族/拼音前缀时，姓名直接位于“班级:”标签前。"""
        raw = "查干其其格 班级:九班 考场:341班 座位:25"
        fields, _ = parse_personal_info_block(raw)
        self.assertEqual("查干其其格", fields.get("name"))

    def test_name_heuristic_allows_long_name(self):
        """无标签姓名（≥5 字，如蒙古族名）不应被 2-4 字限制丢弃。"""
        raw = "座位号:8 考场:341班\n查干其其格"
        fields, _ = parse_personal_info_block(raw)
        self.assertEqual("查干其其格", fields.get("name"))

    def test_name_heuristic_with_middle_dot(self):
        """带间隔号的复姓名（孛尔只斤·特木尔）也应识别。"""
        raw = "座位号:8 考场:341班\n孛尔只斤·特木尔"
        fields, _ = parse_personal_info_block(raw)
        self.assertEqual("孛尔只斤·特木尔", fields.get("name"))

    def test_name_from_school_prefixed_printed_label(self):
        """打印条码标签为“学校 姓名 班级:…”格式时，应取末段姓名（用户实况：东转一中 嘎尔帝）。"""
        raw = "姓名:尔帝 座位号:27\n东转一中 嘎尔帝 班级:八班802200237 考场:341班 座位:27\n802200237"
        fields, _ = parse_personal_info_block(raw)
        self.assertEqual("嘎尔帝", fields.get("name"))

    def test_name_prefers_printed_label_over_wrong_handwriting(self):
        """手写行 OCR 错值（放其泰）不应覆盖打印标签正确值（敖其泰）。"""
        raw = "姓名:放其泰 座位号:21\n东转一中 敖其泰 班级:四班802200231 考场:341班 座位:21\n802200231"
        fields, _ = parse_personal_info_block(raw)
        self.assertEqual("敖其泰", fields.get("name"))

    def test_school_name_not_taken_as_name_when_name_missing(self):
        """打印标签姓名段丢失时，不能把学校名（东转一中）当作姓名。"""
        raw = "东转一中 班级:三班802200233 考场:341班 座位:23\n802200233"
        fields, _ = parse_personal_info_block(raw)
        self.assertNotEqual("东转一中", fields.get("name"))
        self.assertEqual("东转一中", fields.get("school"))

    def test_printed_label_name_via_room_anchor_when_class_label_missing(self):
        """“班级”字样被 OCR 漏识别时，应通过“考场”锚点提取印刷体姓名（且压过手写误识行）。"""
        raw = "姓名:放其泰\n东转一中 敖其泰 考场:341班 座位:21\n802200231"
        fields, _ = parse_personal_info_block(raw)
        self.assertEqual("敖其泰", fields.get("name"))

    def test_printed_label_name_via_seat_anchor(self):
        """只有“座位”锚点时也应能提取印刷体姓名。"""
        raw = "乌日汗 座位:22\n802200232"
        fields, _ = parse_personal_info_block(raw)
        self.assertEqual("乌日汗", fields.get("name"))

    def test_printed_label_anchor_does_not_take_class_word_as_name(self):
        """姓名行缺失时，锚点前的班级词（九班）不能被误作姓名。"""
        raw = "东转一中 班级:九班802200235 考场:341班 座位:25\n802200235"
        fields, _ = parse_personal_info_block(raw)
        self.assertNotEqual("九班", fields.get("name"))
        self.assertIsNone(fields.get("name"))

    def test_marked_printed_name_has_highest_priority(self):
        """条码锚定标签条注入的“印刷体姓名”标记优先级最高，压过手写误识与锚点误读。"""
        # 用户实况：锚点被 OCR 误读（斑级/老场），手写行误识“乌达术”，标签条标记“乌达木”
        raw = ("鸦一中\n乌达术\n斑级：四班\n802200236考场：341班座位：26\n802200236\n"
               "东鸟一中\n乌达木\n老场：341班\n座位26\n印刷体姓名:乌达木")
        fields, _ = parse_personal_info_block(raw)
        self.assertEqual("乌达木", fields.get("name"))
        self.assertEqual("802200236", fields.get("exam_no"))
        self.assertEqual("26", fields.get("seat"))

    def test_roster_match_corrects_ocr_typo(self):
        """花名册裁决：候选“放其泰”应纠正为名单中的“敖其泰”。"""
        raw = "鸦一中\n放其泰\n斑级:四班\n802200231考场:341班座位:21\n印刷体姓名:放其泰"
        fields, _ = parse_personal_info_block(
            raw,
            extra_name_candidates=["放其泰", "效其焱"],
            candidate_names=["敖其泰", "乌达木", "乌日汗"],
        )
        self.assertEqual("敖其泰", fields.get("name"))

    def test_roster_match_prefers_exact_candidate(self):
        """多个候选中有一个与名单完全一致时，直接命中。"""
        raw = "一中\n尔帝\n班级:八班\n802200237\n考场:341班座位:27"
        fields, _ = parse_personal_info_block(
            raw,
            extra_name_candidates=["尔帝", "嘎尔帝"],
            candidate_names=["嘎尔帝"],
        )
        self.assertEqual("嘎尔帝", fields.get("name"))

    def test_roster_match_rejects_low_similarity(self):
        """相似度过低时不纠正，保留原识别结果（避免误配到其他考生）。"""
        raw = "东转一中\n班级:十班\n802200229考场:341班座位:19\n印刷体姓名:余日"
        fields, _ = parse_personal_info_block(
            raw,
            extra_name_candidates=["余日"],
            candidate_names=["奈日", "乌达木"],
        )
        self.assertEqual("余日", fields.get("name"))

    def test_barcode_marker_exam_no_priority(self):
        """条码考号标记优先于手写考号误识（80220G229），且不被更长数字串覆盖。"""
        raw = "考生号:80220G229\n班级:十班\n8022002291\n印刷体姓名:奈日\n条码考号:802200229"
        fields, _ = parse_personal_info_block(raw)
        self.assertEqual("802200229", fields.get("exam_no"))

    def test_seat_not_swallow_exam_no(self):
        """座位号标签后换行紧跟考号长数字时，不能把考号当成座位号。"""
        raw = "座位号：\n802200211\n考场：341班\n座位：1"
        fields, _ = parse_personal_info_block(raw)
        self.assertEqual("1", fields.get("seat"))
        self.assertEqual("341", fields.get("room"))


if __name__ == "__main__":
    unittest.main()
