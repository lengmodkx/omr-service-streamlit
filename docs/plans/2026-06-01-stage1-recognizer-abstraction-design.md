# 阶段 1 详细设计 — Recognizer 协议抽象

> 文档状态：待评审
> 实施分支：`feat/recognizer-abstraction`（从 `master` 切出，含已合入的阶段 2 成果）
> 关联文档：
> - [优化方案总览 §3.1 适配器模式](../AI-Marker-Suite借鉴与架构优化方案.md)
> - [阶段 2 ScoreCalculator 集中化设计](2026-06-01-stage2-score-calculator-design.md)
> 预计工时：~1.5 天

---

## 一、目标

把当前散落在 [omr_demo/core/golden_template.py:206](../omr_demo/core/golden_template.py#L206) 和 [omr_demo/core/processor.py:108](../omr_demo/core/processor.py#L108) 的两种识别方式（黄金模板法 / 差分法），**统一到 `Recognizer` 协议下**。

**核心收益**：
1. 新增识别器（如 YOLO / AI 评分）只需新增一个适配器类，**零侵入 `app.py`**
2. 双识别器交叉验证（阶段 7）成为可能——只需在 `RecognizeManager` 遍历所有 `Recognizer`
3. 单元测试可对每个适配器独立验证输出结构

---

## 二、当前现状

| 识别器 | 位置 | 返回结构 | 是否符合统一目标 |
|--------|------|---------|----------------|
| **黄金模板法** | `golden_template.py:206-353` | `{answers, total, empty_count, multi_count, card_flag, debug_lines}` + 单题内 `correct` 字段 | ⭐⭐⭐ 接近但缺 `duration_ms` / `recognizer_id` |
| **差分法** | `processor.py:108-179` | `{q: str|None}`（仅答案，状态从字符串后缀"多涂"推断）| ⭐ 需重整 |
| **手动区域识别** | `processor.py:209-278` | 同差分法，区域限定 | ⭐ 需重整 |
| **自定义选项框** | `processor.py:280-332` | 同差分法，自定义坐标 | ⭐ 需重整 |

**统一目标**：所有识别器返回 `RecognizeResult`（`core/recognizer.py` 定义）。

---

## 三、文件改动清单

| 文件 | 操作 | 估计行数 | 改动说明 |
|------|------|---------|---------|
| `omr_demo/core/recognizer.py` | **新增** | ~120 行 | 协议 + 上下文 + 结果 dataclass + 工厂函数 |
| `omr_demo/core/recognizers/__init__.py` | **新增** | ~15 行 | 暴露 `DifferentialRecognizer` / `GoldenTemplateRecognizer` |
| `omr_demo/core/recognizers/differential.py` | **新增** | ~80 行 | 适配器：包装 `CardProcessor.recognize_choices` |
| `omr_demo/core/recognizers/golden.py` | **新增** | ~60 行 | 适配器：包装 `GoldenTemplate.recognize`（薄包装） |
| `omr_demo/test_recognizer.py` | **新增** | ~180 行 | 协议一致性 + 字段完备性 + 跨识别器结果对比 |
| `omr_demo/app.py` | **改** | -3 行（`gtp.recognize(...)` → `recognizer.recognize(...)`），+5 行（构造 adapter） | Tab2 走协议入口；`gtp.image/bubbles` 预览图生成仍直接访问 |
| `omr_demo/core/processor.py` | **不改** | 0 | 内部方法不改造，外部通过 DifferentialRecognizer 包装 |
| `omr_demo/core/golden_template.py` | **不改** | 0 | 同上 |

**关键决策**：
- ✅ **包装**而非重写：内部识别逻辑 0 改动
- ✅ 协议字段补齐 `duration_ms` / `recognizer_id`（为阶段 7 交叉验证铺垫）
- ❌ **不动 `process_pair()`**（它做太多事：barcode + 裁剪 + 手动区域，阶段 5 拆 UI 时一起处理）
- ❌ **Tab2 预览图**仍直接访问 `gtp.image` / `gtp.bubbles`（demo 阶段不强行隔离）

---

## 四、API 详细签名

### 4.1 `core/recognizer.py`

```python
from typing import Protocol, runtime_checkable, Optional
from dataclasses import dataclass, field
import time
import numpy as np


@dataclass
class RecognizeContext:
    """识别上下文 — 识别器所需的所有外部依赖打包"""
    template_config: Optional[dict] = None      # 模板 JSON dict
    blank_refs: dict[str, np.ndarray] = field(default_factory=dict)  # 空白参考 {A, B}
    column_boxes: list[dict] = field(default_factory=list)            # 黄金模板列框
    custom_bubbles: list[dict] = field(default_factory=list)         # 自定义选项框
    standard_answers: dict[int, str] = field(default_factory=dict)   # 标准答案（可选）
    page: str = "A"                             # A 面 / B 面
    threshold: float = 0.15                     # 识别阈值（差分法）


@dataclass
class RecognizeResult:
    """统一识别结果 — 所有识别器都返回这个结构"""
    answers: dict[int, dict]                    # {q: {"answer": str|None, "status": str, "correct": bool|None}}
    total: int
    empty_count: int
    multi_count: int
    card_flag: Optional[str] = None             # "abnormal" / "suspicious_blank" / None
    debug_lines: list[str] = field(default_factory=list)
    duration_ms: float = 0.0                    # 识别耗时（毫秒）
    recognizer_id: str = ""                     # 哪个识别器跑的


@runtime_checkable
class Recognizer(Protocol):
    """识别器协议 — 任意识别方式都实现此接口"""
    name: str                                   # 中文名："黄金模板对比法" / "差分法"
    id: str                                     # 唯一 ID："golden" / "differential" / "yolo" / "ai"
    requires_blank: bool                        # 是否需要空白参考
    
    def can_handle(self, ctx: RecognizeContext) -> bool:
        """根据上下文判断是否可处理（如 YOLO 需看置信度，未配置列框则不可处理）"""
        ...
    
    def recognize(self, image: np.ndarray, ctx: RecognizeContext) -> RecognizeResult:
        """核心识别逻辑"""
        ...


def make_recognizer(recognizer_id: str, **kwargs) -> Recognizer:
    """工厂函数 — 根据 ID 构造识别器实例
    
    用法:
        rec = make_recognizer("golden", golden_template=gtp)
        rec = make_recognizer("differential", processor=cp, page="A")
    """
    if recognizer_id == "golden":
        from core.recognizers.golden import GoldenTemplateRecognizer
        return GoldenTemplateRecognizer(**kwargs)
    if recognizer_id == "differential":
        from core.recognizers.differential import DifferentialRecognizer
        return DifferentialRecognizer(**kwargs)
    raise ValueError(f"Unknown recognizer_id: {recognizer_id}")
```

### 4.2 `core/recognizers/golden.py`

```python
class GoldenTemplateRecognizer:
    """黄金模板对比法适配器 — 包装 GoldenTemplate.recognize()"""
    
    name = "黄金模板对比法"
    id = "golden"
    requires_blank = False  # 用黄金模板自身作参考，不需要空白
    
    def __init__(self, golden_template: GoldenTemplate):
        self._gtp = golden_template
    
    def can_handle(self, ctx: RecognizeContext) -> bool:
        return self._gtp is not None and len(self._gtp.bubbles) > 0
    
    def recognize(self, image: np.ndarray, ctx: RecognizeContext) -> RecognizeResult:
        start = time.perf_counter()
        result_dict = self._gtp.recognize(image, debug=bool(ctx.standard_answers))
        duration = (time.perf_counter() - start) * 1000
        return RecognizeResult(
            answers=result_dict["answers"],
            total=result_dict["total"],
            empty_count=result_dict["empty_count"],
            multi_count=result_dict["multi_count"],
            card_flag=result_dict.get("card_flag"),
            debug_lines=result_dict.get("debug_lines", []),
            duration_ms=round(duration, 2),
            recognizer_id=self.id,
        )
```

### 4.3 `core/recognizers/differential.py`

```python
class DifferentialRecognizer:
    """差分法适配器 — 包装 CardProcessor.recognize_choices()"""
    
    name = "差分法"
    id = "differential"
    requires_blank = True  # 需要空白参考
    
    def __init__(self, processor: CardProcessor, page: str = "A"):
        self._proc = processor
        self._page = page
    
    def can_handle(self, ctx: RecognizeContext) -> bool:
        return self._proc is not None and len(self._proc.template["pages"][self._page].get("bubbles", [])) > 0
    
    def recognize(self, image: np.ndarray, ctx: RecognizeContext) -> RecognizeResult:
        start = time.perf_counter()
        raw = self._proc.recognize_choices(image, page=self._page, threshold=ctx.threshold)
        duration = (time.perf_counter() - start) * 1000
        
        answers = {}
        empty_count = 0
        multi_count = 0
        for q, ans in raw.items():
            if ans is None:
                answers[q] = {"answer": None, "status": "empty", "correct": None}
                empty_count += 1
            elif str(ans).endswith("(多涂)"):
                answers[q] = {"answer": str(ans)[:-4], "status": "multi", "correct": None}
                multi_count += 1
            else:
                answers[q] = {"answer": ans, "status": "single", "correct": None}
        
        return RecognizeResult(
            answers=answers,
            total=len(raw),
            empty_count=empty_count,
            multi_count=multi_count,
            card_flag=None,
            debug_lines=[],
            duration_ms=round(duration, 2),
            recognizer_id=self.id,
        )
```

---

## 五、调用点迁移细节

### 5.1 改 `app.py:588`（commit 3）

**原代码**：
```python
gtp = st.session_state.golden_template
# ...
r = gtp.recognize(img_a, debug=debug_mode)
r["_key"] = key
r["_file_a"] = file_a.name
r["_file_b"] = file_b.name if file_b else ""
```

**新代码**：
```python
gtp = st.session_state.golden_template
recognizer = make_recognizer("golden", golden_template=gtp)
# ...
result = recognizer.recognize(img_a, RecognizeContext(standard_answers=st.session_state.standard_answers))
# 转换为 app.py 期望的 dict 形态（保持下游代码 0 改动）
r = {
    "answers": {q: info for q, info in result.answers.items()},
    "total": result.total,
    "empty_count": result.empty_count,
    "multi_count": result.multi_count,
    "card_flag": result.card_flag,
    "debug_lines": result.debug_lines,
    "_key": key,
    "_file_a": file_a.name,
    "_file_b": file_b.name if file_b else "",
}
```

**关键差异**：
- `r` 仍然是 dict（保持下游 `r["answers"]` / `r["total"]` 等访问 0 改动）
- 识别耗时通过 `result.duration_ms` 可获取（demo 暂不展示，预留扩展点）
- 后续若加"识别器选择下拉框"，仅需替换 `make_recognizer(...)` 即可，零侵入

### 5.2 `app.py:596-619` 预览图生成 **不动**

`gtp.image` / `gtp.bubbles` 的访问是**黄金模板特有的**（用于在原图上画气泡点），不属于 `Recognizer` 协议关注点。阶段 7 交叉验证时再考虑把 `visualize()` 加到协议上。

---

## 六、测试设计

### 6.1 `test_recognizer.py`（~180 行）

**测试矩阵**：

| 测试 | 类型 | 目的 |
|------|------|------|
| `test_recognizer_protocol_conformance` | 协议一致性 | `isinstance(rec, Recognizer)` 通过 |
| `test_recognize_result_required_fields` | 字段完备性 | 所有 Recognizer 返回的 result 含 8 个必需字段 |
| `test_recognize_result_default_values` | 字段完备性 | 缺省值合理（debug_lines=[]、duration_ms=0.0 等） |
| `test_can_handle_returns_bool` | 协议一致性 | can_handle 返回 bool |
| `test_recognize_returns_dataclass` | 类型检查 | 返回 `RecognizeResult` 实例 |
| `test_duration_ms_positive` | 行为 | 真实识别耗时 > 0 |
| `test_recognizer_id_matches_id` | 行为 | result.recognizer_id == self.id |
| `test_make_recognizer_factory` | API | `make_recognizer("golden", ...)` / `make_recognizer("differential", ...)` 正确 |
| `test_make_recognizer_unknown_id` | API | 未知 ID 抛 `ValueError` |
| `test_differential_empty_image` | 行为 | 空图像不崩溃，empty_count = total |
| `test_differential_multi_detection` | 行为 | "(多涂)" 字符串 → status=multi |
| `test_golden_invalid_image` | 行为 | 无效图像返回 card_flag="invalid_image" |

**总计 12 个测试函数**。

### 6.2 真实图像测试（不依赖 fixtures）

为了测试真实识别行为，构造最小测试图像：
- 3×3 像素 + 噪声 → 模拟空白卡
- 不实际跑识别（避免依赖 cv2 大图），改用 mock 替换 `_gtp.recognize()` / `_proc.recognize_choices()`

---

## 七、实施 checklist

### Commit 1: `feat(recognizer): 新增 Recognizer 协议 + 工厂函数`

- [ ] 新建 `omr_demo/core/recognizer.py`，120 行
- [ ] **验证**：`python -c "from core.recognizer import Recognizer, RecognizeContext, RecognizeResult, make_recognizer"` 无报错

### Commit 2: `feat(recognizer): GoldenTemplateRecognizer 适配器`

- [ ] 新建 `omr_demo/core/recognizers/__init__.py`（15 行）
- [ ] 新建 `omr_demo/core/recognizers/golden.py`（60 行）
- [ ] **验证**：手动 `GoldenTemplateRecognizer(gtp).recognize(test_img, ctx)` 返回 RecognizeResult

### Commit 3: `feat(recognizer): DifferentialRecognizer 适配器`

- [ ] 新建 `omr_demo/core/recognizers/differential.py`（80 行）
- [ ] **验证**：手动 `DifferentialRecognizer(cp, "A").recognize(test_img, ctx)` 返回 RecognizeResult

### Commit 4: `test(recognizer): 协议一致性与字段完备性测试`

- [ ] 新建 `omr_demo/test_recognizer.py`（180 行，12 测试函数）
- [ ] **验证**：`python omr_demo/test_recognizer.py` → 12 passed, 0 failed

### Commit 5: `refactor(app): Tab2 走 Recognizer 协议入口`

- [ ] 改 `omr_demo/app.py:588`（`gtp.recognize` → `recognizer.recognize`）
- [ ] 顶部 import 增加 `from core.recognizer import make_recognizer, RecognizeContext`
- [ ] **验证**：跑 `streamlit run app.py`，Tab2 上传 1 张卡，识别结果数字与改造前完全一致

---

## 八、风险与回滚

| 风险 | 触发条件 | 兜底 |
|------|---------|------|
| **识别结果回退** | `DifferentialRecognizer.recognize()` 与原 `recognize_choices()` 输出不一致 | 用单卡对比新旧两次输出，回退则 `git revert commit 3` |
| **Tab2 跑批失败** | `result.recognize()` 抛异常或返回结构错误 | `git revert commit 5`，`app.py` 回到 `gtp.recognize` 直接调用 |
| **协议字段定义偏差** | 阶段 7 交叉验证时发现 RecognizeResult 缺关键字段 | 在 `RecognizeResult` 加字段，向后兼容（默认值），已实现的适配器无需改动 |
| **顺手改识别逻辑** | 包装时"顺带优化"识别算法 | **不做**，记 issue 留到独立 PR |

---

## 九、验收清单（5 项硬指标）

- [ ] ① `omr_demo/core/recognizer.py` 文件存在，定义 `Recognizer` Protocol / `RecognizeContext` / `RecognizeResult` / `make_recognizer`
- [ ] ② `omr_demo/core/recognizers/` 包存在，含 `golden.py` 和 `differential.py`
- [ ] ③ `python omr_demo/test_recognizer.py` → 12 passed, 0 failed
- [ ] ④ `grep "gtp.recognize" omr_demo/app.py` 输出 0 行（Tab2 走协议入口）
- [ ] ⑤ Tab2 上传 1 张卡跑通，识别结果数字与改造前完全一致

**5 项全过 = 阶段 1 完工，进阶段 7（双识别器交叉验证）**。

---

## 十、相关文档

- [优化方案总览 §3.1 适配器模式识别器抽象](../AI-Marker-Suite借鉴与架构优化方案.md)
- [优化方案总览 §五 阶段 1 → 2 → 7 demo 阶段路径](../AI-Marker-Suite借鉴与架构优化方案.md)
- [阶段 2 ScoreCalculator 集中化设计](2026-06-01-stage2-score-calculator-design.md)
- 关联文件：
  - [omr_demo/app.py](../../omr_demo/app.py)（改 Tab2 入口）
  - [omr_demo/core/golden_template.py](../../omr_demo/core/golden_template.py)（不改，被包装）
  - [omr_demo/core/processor.py](../../omr_demo/core/processor.py)（不改，被包装）
