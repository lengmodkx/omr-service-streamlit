# 阶段 7 详细设计 — 双识别器交叉验证

> 文档状态：待评审
> 实施分支：`feat/cross-validation`（从 `feat/recognizer-abstraction` 切出，含阶段 1+2 全部成果）
> 关联文档：
> - [优化方案总览 §3.5 双识别器交叉验证](../AI-Marker-Suite借鉴与架构优化方案.md)
> - [阶段 1 Recognizer 协议抽象设计](2026-06-01-stage1-recognizer-abstraction-design.md)
> - [阶段 2 ScoreCalculator 集中化设计](2026-06-01-stage2-score-calculator-design.md)
> 预计工时：~1.5 天

---

## 一、目标

在阶段 1 落地的 `Recognizer` 协议基础上，引入 `RecognizerManager` 对**同一张卡**跑**多个识别器**，对比结果并标记分歧题目，让低置信度题目自动进入人工核对面板。

**核心收益**：
1. **降低单识别器误判的隐性风险**——黄金模板的边界案例（涂改/极浅填涂）能被差分法兜底
2. **为 Tab3 人工核对分流**——分歧题自动进高优先级列表，正常题不打扰老师
3. **为未来 AI 评分做架构铺垫**——`RecognizerManager` 天然支持 N 个识别器扩列（YOLO/AI 评估/规则引擎等）

**OMR 场景对应**：用户常怀疑"这套模板和这张答题卡匹配吗？"——`RecognizerManager.cross_validate()` 直接给出量化回答。

---

## 二、当前现状

| 识别器 | 阶段 1 状态 |
|--------|------------|
| 黄金模板法 | ✅ 已包成 `GoldenTemplateRecognizer` |
| 差分法 | ✅ 已包成 `DifferentialRecognizer` |
| Tab2 跑批 | 只调 1 个识别器,无交叉验证 |

**痛点**:Tab3 人工核对时,老师需逐题判断"识别器对不对",效率低。

---

## 三、文件改动清单

| 文件 | 操作 | 估计行数 | 改动说明 |
|------|------|---------|---------|
| `omr_demo/core/recognizer.py` | **改** | +20 行 | 新增 `CrossValidatedResult` dataclass(继承 `RecognizeResult`) |
| `omr_demo/core/recognizer_manager.py` | **新增** | ~150 行 | `RecognizerManager` 类 + `_compare_answers()` 辅助 |
| `omr_demo/test_recognizer_manager.py` | **新增** | ~180 行 | 12+ 测试函数,覆盖 4 种 agreement_type |
| `omr_demo/app.py` | **改** | +30 行 | Tab2 加"启用交叉验证"复选框;Tab3 高亮分歧题 |
| `omr_demo/core/recognizers/*.py` | **不改** | 0 | 已有适配器无需修改 |

**关键决策**:
- ✅ **CrossValidatedResult 继承 RecognizeResult** — 下游代码(scoring、preview)无需改类型判断
- ✅ **N 个识别器** — manager 通用,不限于 2 个
- ❌ **不并行** — 顺序执行(2 个识别器 ~200ms,用户感知不强)
- ❌ **不持久化** — 交叉验证结果跟随本次 session_state,刷新即丢
- ❌ **不动 `process_pair()`** — 阶段 5 拆 UI 时一起处理

---

## 四、API 详细签名

### 4.1 `core/recognizer.py` — 新增 dataclass

```python
@dataclass
class CrossValidatedResult(RecognizeResult):
    """交叉验证结果 — 继承 RecognizeResult,新增验证字段

    字段:
        per_q_cv: {q: {per_recognizer, agreed, consensus, agreement_type}}
        agreement_rate: float                # 全题"agreed=True"占比 (0.0~1.0)
        disputed_questions: list[int]        # 分歧题号列表(供 Tab3 高亮)
        recognizer_results: dict             # 各识别器独立结果 {id: RecognizeResult}
    """
    per_q_cv: dict = field(default_factory=dict)
    agreement_rate: float = 1.0
    disputed_questions: list = field(default_factory=list)
    recognizer_results: dict = field(default_factory=dict)
```

### 4.2 `core/recognizer_manager.py`

```python
class RecognizerManager:
    """识别器管理器 — 持有 N 个识别器,提供交叉验证能力"""

    def __init__(self, recognizers: list):
        """
        Args:
            recognizers: list[Recognizer] 至少 1 个,顺序即优先级
        """
        if not recognizers:
            raise ValueError("recognizers 不能为空")
        self._recognizers = list(recognizers)

    @property
    def recognizers(self) -> list:
        return list(self._recognizers)

    def add(self, recognizer) -> None:
        """动态注册新识别器(阶段 9 留扩展点)"""
        self._recognizers.append(recognizer)

    def cross_validate(self, image: np.ndarray, ctx: RecognizeContext) -> CrossValidatedResult:
        """
        同一张卡跑所有识别器,对比每题结果

        Returns:
            CrossValidatedResult 实例,含:
            - answers: 取"全匹配"题目的共识;分歧题取 None + status="uncertain"
            - per_q_cv: 每题详细验证结果
            - disputed_questions: 分歧题号列表
            - agreement_rate: 全题 agreed 占比
        """

    @staticmethod
    def _compare_answers(per_recognizer: dict) -> dict:
        """
        对比 N 个识别器对同一题的答案

        Args:
            per_recognizer: {recognizer_id: answer_str_or_None}

        Returns:
            {
                "agreed": bool,
                "consensus": str | None,        # 共识答案(分歧时为 None)
                "agreement_type": str,           # "all_empty" / "all_match" / "one_uncertain" / "disputed"
            }
        """
```

### 4.3 `_compare_answers` 算法

| 情形 | agreement_type | agreed | consensus |
|------|---------------|--------|-----------|
| 全空(None) | `all_empty` | True | None |
| 1 个非空 + 其它空 | `one_uncertain` | False | 非空那个 |
| 全非空且集合相等 | `all_match` | True | 第一个 |
| 全非空且集合不等 | `disputed` | False | None |
| 1 空 + N 非空且 N 一致 | `all_match` | True | 一致的那个 |
| 1 空 + N 非空且 N 不一致 | `disputed` | False | None |

**多选处理**:用 `set(answer)` 比较,顺序无关。

---

## 五、Tab2 调用改造

### 5.1 新增复选框(Tab2 顶部)

```python
enable_cv = st.checkbox(
    "启用双识别器交叉验证 (黄金模板 + 差分法, 慢但更准)",
    value=False, key="enable_cross_validate",
    help="同一张卡跑两个识别器,分歧题自动进人工核对面板"
)
```

### 5.2 改造 `_process_single`

**原代码**:
```python
recognizer = make_recognizer("golden", golden_template=gtp)
result = recognizer.recognize(img_a, RecognizeContext(...))
r = result.to_legacy_dict()
```

**新代码**:
```python
if enable_cv:
    # 双识别器交叉验证
    golden_rec = make_recognizer("golden", golden_template=gtp)
    diff_rec = make_recognizer("differential", processor=st.session_state.processor, page="A")
    manager = RecognizerManager([golden_rec, diff_rec])
    cv_result = manager.cross_validate(img_a, RecognizeContext(
        standard_answers=st.session_state.standard_answers
    ))
    r = cv_result.to_legacy_dict()
    r["_disputed_questions"] = cv_result.disputed_questions
    r["_agreement_rate"] = cv_result.agreement_rate
else:
    # 单识别器(原有路径)
    recognizer = make_recognizer("golden", golden_template=gtp)
    result = recognizer.recognize(img_a, RecognizeContext(...))
    r = result.to_legacy_dict()
    r["_disputed_questions"] = []
    r["_agreement_rate"] = 1.0
```

**关键点**:
- `cv_result.to_legacy_dict()` 继承自 `RecognizeResult`,下游代码 0 改动
- 新增字段 `_disputed_questions` 和 `_agreement_rate` 供 Tab3 使用
- `enable_cv=False` 时,完全走原路径,0 性能损耗

---

## 六、Tab3 展示改造

### 6.1 单卡详情页(现有 `st.dataframe` 区域)

在每张卡的详情表格**标题旁**加分歧题数标记:

```python
disputed = r.get("_disputed_questions", [])
if disputed:
    st.warning(f"⚠️ 本卡有 {len(disputed)} 个分歧题: {disputed[:5]}{'...' if len(disputed)>5 else ''}")
    st.caption("建议优先人工核对分歧题")
```

### 6.2 摘要表(原有 rows 列表)加列

```python
rows.append({
    # ... 现有字段 ...
    "分歧": len(r.get("_disputed_questions", [])),
    "识别器一致率": f"{r.get('_agreement_rate', 1.0)*100:.0f}%",
})
```

---

## 七、测试设计

### 7.1 `test_recognizer_manager.py` — 12+ 测试函数

| 测试 | 目的 |
|------|------|
| `test_compare_all_empty` | 全 None → all_empty |
| `test_compare_all_match_single` | 全 "A" → all_match |
| `test_compare_all_match_multi` | "ABC" / "BCA" / "CAB" → all_match (set) |
| `test_compare_one_uncertain_empty_diff` | None vs "A" → one_uncertain, consensus="A" |
| `test_compare_one_uncertain_diff_empty` | "A" vs None → one_uncertain, consensus="A" |
| `test_compare_disputed_single` | "A" vs "B" → disputed, consensus=None |
| `test_compare_disputed_multi` | "AB" vs "ABC" → disputed |
| `test_manager_empty_recognizers_raises` | 空列表 → ValueError |
| `test_manager_single_recognizer` | 1 个识别器,所有题都应 all_match |
| `test_manager_two_agree_all` | 2 个 mock 识别器输出相同 → agreement_rate=1.0 |
| `test_manager_two_dispute_some` | 2 个 mock 识别器部分分歧 → agreement_rate<1.0, disputed_questions 非空 |
| `test_manager_to_legacy_dict` | cross_validate() 返回值可转 legacy dict |
| `test_manager_per_q_cv_format` | per_q_cv 字段含 per_recognizer/agreed/agreement_type |

**总计 13 个测试函数**(覆盖 4 种 agreement_type + N=1/2 边界)。

### 7.2 Mock 策略

不依赖真实图像,用 `unittest.mock.MagicMock` 替换底层识别方法:

```python
def make_mock_recognizer(rec_id: str, answers_map: dict):
    """构造一个 mock 识别器,返回预设的 answers_map"""
    rec = MagicMock()
    rec.id = rec_id
    rec.name = f"Mock-{rec_id}"
    rec.requires_blank = False
    rec.recognize = MagicMock(return_value=RecognizeResult(
        answers={q: {"answer": a, "status": "single" if a else "empty", "correct": None}
                 for q, a in answers_map.items()},
        total=len(answers_map),
        empty_count=sum(1 for a in answers_map.values() if a is None),
        multi_count=0,
        recognizer_id=rec_id,
    ))
    return rec
```

---

## 八、实施 checklist

### Commit 1: `feat(cv): RecognizerManager + CrossValidatedResult`

- [ ] 改 `omr_demo/core/recognizer.py`,新增 `CrossValidatedResult` dataclass
- [ ] 新建 `omr_demo/core/recognizer_manager.py`(150 行)
- [ ] **验证**:`python -c "from core.recognizer_manager import RecognizerManager"` 无报错

### Commit 2: `test(cv): 交叉验证逻辑测试 (4 种 agreement_type 全覆盖)`

- [ ] 新建 `omr_demo/test_recognizer_manager.py`(180 行,13 测试函数)
- [ ] **验证**:`python omr_demo/test_recognizer_manager.py` → 13 passed, 0 failed

### Commit 3: `feat(app): Tab2 加交叉验证开关 + Tab3 高亮分歧题`

- [ ] 改 `omr_demo/app.py` Tab2(`_process_single` 函数 + 顶部 checkbox)
- [ ] 改 `omr_demo/app.py` Tab3(单卡详情页 + 摘要表)
- [ ] **验证**:启动 `streamlit run app.py`,Tab2 勾选/不勾选交叉验证,跑 1 张卡,数字与改造前一致

---

## 九、风险与回滚

| 风险 | 触发条件 | 兜底 |
|------|---------|------|
| **交叉验证拖慢批处理** | 2 个识别器 ~400ms/张 | 复选框默认关闭;Tab2 显示耗时统计(可选) |
| **mock 不真实** | 测试通过但真实场景下 _compare_answers 逻辑边界 case 漏 | commit 3 必跑人工验证;如有遗漏,补丁直接加到 `_compare_answers` |
| **Tab2 跑批异常** | `cv_result` 字段缺失导致下游崩溃 | 验证 `to_legacy_dict()` 继承链,所有下游访问走 legacy dict |
| **顺手优化识别逻辑** | 包装时"顺带优化"识别算法 | **不做**,记 issue 留到独立 PR |
| **Card flag 冲突** | existing `card_flag="abnormal"` 与新的 "uncertain" 冲突 | 优先级:`invalid_image` > `disputed` > `abnormal` > `suspicious_blank` > None |

---

## 十、验收清单(5 项硬指标)

- [ ] ① `omr_demo/core/recognizer_manager.py` 文件存在,定义 `RecognizerManager` 类
- [ ] ② `CrossValidatedResult` dataclass 在 `recognizer.py` 中定义,继承 `RecognizeResult`
- [ ] ③ `python omr_demo/test_recognizer_manager.py` → 13 passed, 0 failed
- [ ] ④ `grep "enable_cross_validate" omr_demo/app.py` 至少 1 行(Tab2 复选框)
- [ ] ⑤ Tab2 关闭交叉验证时,识别结果数字与阶段 1 改造前完全一致(0 回归)

**5 项全过 = 阶段 7 完工,demo 阶段 1+2+7 全闭环。**

---

## 十一、Demo 阶段全闭环

| 阶段 | 状态 | 关键产出 |
|------|------|---------|
| **1** Recognizer 抽象 | ✅ 4/5 自动 | 协议 + 2 适配器 + 12 测试 |
| **2** ScoreCalculator | ✅ 5/6 自动 | 集中算分 + 22 单元 + 15 回归 |
| **7** 交叉验证 | ⏳ 启动中 | Manager + 13 测试 + Tab2 开关 |
| **人工验证** | ⏳ 3 项 | 阶段 1 ⑤ / 阶段 2 ⑥ / 阶段 7 ⑤ |

**3 个未完成的人工验证合并到 1 次 streamlit 跑批验证**——3 项改造都跑同一张测试卡,数字应与改造前完全一致(交叉验证默认关闭,不影响单识别器路径)。

---

## 十二、相关文档

- [优化方案总览 §3.5 双识别器交叉验证](../AI-Marker-Suite借鉴与架构优化方案.md)
- [优化方案总览 §五 阶段 1 → 2 → 7 demo 阶段路径](../AI-Marker-Suite借鉴与架构优化方案.md)
- [阶段 1 Recognizer 协议抽象设计](2026-06-01-stage1-recognizer-abstraction-design.md)
- [阶段 2 ScoreCalculator 集中化设计](2026-06-01-stage2-score-calculator-design.md)
- 关联文件：
  - [omr_demo/app.py](../../omr_demo/app.py)（改 Tab2 入口 + Tab3 高亮）
  - [omr_demo/core/recognizer.py](../../omr_demo/core/recognizer.py)（加 CrossValidatedResult）
  - [omr_demo/core/recognizers/](../../omr_demo/core/recognizers/)（不改,被 Manager 调用）
