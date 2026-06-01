# 阶段 2 详细设计 — ScoreCalculator 集中化

> 文档状态：已评审，待实施
> 实施分支：`feat/score-calculator`（从 master 切出）
> 关联文档：[优化方案总览](../AI-Marker-Suite借鉴与架构优化方案.md)（§3.3）
> 预计工时：~6.5 小时（1 个工作日）

---

## 一、目标

把散落在 `app.py:810-825` 的内联算分循环、散落在 `golden_template.py:322-326` 的多选集合匹配，**全部抽离**到独立的 `core/score_calculator.py` 模块。

**核心收益**：
1. 算分逻辑 100% 可测试（纯函数，无 Streamlit 依赖）
2. 微服务化时此模块可**零改动平移**到 FastAPI/gRPC 后端
3. 未来业务规则变更（"多选少选给一半分""0.5 取整"）只改一个文件

---

## 二、文件改动清单

| 文件 | 操作 | 估计行数 |
|------|------|---------|
| `omr_demo/core/score_calculator.py` | **新增** | ~80 行（含 docstring） |
| `omr_demo/tests/__init__.py` | **新增** | 空文件 |
| `omr_demo/tests/test_score_calculator.py` | **新增** | ~120 行（22 用例） |
| `omr_demo/app.py` | **改** | -8 行（替换 810-825 内联循环）<br>+2 行（export_rows 加列） |
| `omr_demo/core/golden_template.py` | **不改** | 阶段 2 不动 recognizer |

**`golden_template.py:322-326` 的 `correct` 字段保留**：阶段 2 暂不动它，识别器仍输出 `correct` 字段供显示用。ScoreCalculator 是**平行的"真理源"**，两者并存直到阶段 7 整合。

---

## 三、API 详细签名

### 3.1 `ScoringConfig` dataclass

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class ScoringConfig:
    """算分配置 — 取代散落的 dict 参数"""
    round_step: float = 1.0        # 取整步长（默认 1 = 整数；0.5 = 半取整）
    round_method: str = "round"    # "round" / "floor" / "ceil"
    multi_partial: bool = False    # 多选少选是否给一半分（demo 阶段永远 False）
    ignore_case: bool = True       # 答案大小写不敏感
```

**字段裁剪说明**（vs. 最初的"建议方案"）：
- ❌ 删除 `per_question_score` ——demo 阶段硬编码 1 分
- ❌ 删除 `bonus_*` 字段 ——demo 阶段无勤勉分概念
- ✅ 保留 `multi_partial` ——保留扩展点，但默认关闭

### 3.2 函数签名

```python
def round_score(score: float | None, cfg: ScoringConfig | None = None) -> float | None:
    """取整。None 透传不抛异常。"""


def match_answer(
    student: str | None,
    gold: str | None,
    multi: bool = False,
    cfg: ScoringConfig | None = None,
) -> bool:
    """单选严格匹配 / 多选集合匹配。None/空串 → False。"""


def calc_total_score(
    answers: dict,            # {q: {answer, status}} 来自识别器
    standard: dict,           # {q: "A" 或 "ABC"}     来自用户/标准答案
    cfg: ScoringConfig | None = None,
) -> dict:
    """
    返回:
    {
        "total": float,         # 总分（已取整）
        "total_full": float,    # 满分
        "per_q": [              # 每题明细
            {"q": int, "earned": float, "full": float, "correct": bool, "status": str}
        ],
        "stats": {              # 状态计数
            "single": int, "empty": int, "multi": int, "uncertain": int, "correct": int
        }
    }
    """
```

**实现规则**（与现有 app.py:810-825 行为严格一致）：

| 状态 | earned | correct |
|------|--------|---------|
| `single` + 匹配 | 1.0 | True |
| `single` + 不匹配 | 0.0 | False |
| `empty` | 0.0 | False |
| `multi` | 0.0（按当前 demo 规则） | False |
| `uncertain` | 0.0 | False |
| `None` answer | 0.0 | False |

**多选识别**（`multi` 参数）：当 `len(gold) > 1` 时 `multi=True`，走 `match_answer` 的集合匹配。

---

## 四、22 个单元测试场景

完整场景清单（[优化方案 §3.1 测试策略](AI-Marker-Suite借鉴与架构优化方案.md#31-适配器模式识别器抽象为统一接口--优先级最高) 已列出）。下面给出**测试函数骨架**，实施时按此填写。

### 4.1 `test_round_score`（5 例）

```python
def test_round_score_integer_unchanged(): ...
def test_round_score_normal_rounding(): ...
def test_round_score_half_step(): ...
def test_round_score_negative_half_step(): ...
def test_round_score_none_passthrough(): ...
```

### 4.2 `test_match_answer`（8 例）

```python
def test_single_match_exact(): ...
def test_single_mismatch(): ...
def test_multi_match_exact(): ...
def test_multi_mismatch_missing_one(): ...
def test_multi_mismatch_extra_one(): ...
def test_case_insensitive_match(): ...
def test_case_sensitive_strict_mismatch(): ...
def test_empty_or_none_safe(): ...
```

### 4.3 `test_calc_total_score`（9 例，原 10 例砍掉 #22）

```python
def test_all_correct(): ...
def test_all_wrong(): ...
def test_all_empty(): ...
def test_all_multi_zero_score(): ...
def test_half_correct(): ...
def test_mixed_status(): ...
def test_extra_answers_ignored(): ...
def test_missing_answers_zero_score(): ...
def test_total_rounding_boundary(): ...
```

**总计 22 例**。

---

## 五、调用点迁移细节

### 5.1 改 `app.py:810-825`（commit 2）

**原代码**（[app.py:810-825](../omr_demo/app.py#L810-L825)）：

```python
sc = 0
tot = len(golden_ans)
for q, std_ans in golden_ans.items():
    final_ans = new_corr.get(q)  # 人工修正优先
    if not final_ans:
        ans_info = result["answers"].get(q, {})
        raw = ans_info.get("answer")
        final_ans = raw if raw else ""
    if final_ans:
        if len(std_ans) > 1:
            if set(final_ans) == set(std_ans):
                sc += 1
        elif final_ans == std_ans:
            sc += 1
st.metric("选择题最终得分", f"{sc} / {tot}")
```

**新代码**：

```python
from core.score_calculator import calc_total_score, ScoringConfig

# 构造 answers：人工修正优先于 auto
effective_answers = {}
for q, std_ans in golden_ans.items():
    corrected = new_corr.get(q)
    if corrected:
        effective_answers[q] = {"answer": corrected, "status": "single"}
    else:
        ans_info = result["answers"].get(q, {})
        effective_answers[q] = {
            "answer": ans_info.get("answer"),
            "status": ans_info.get("status", "empty"),
        }

result_score = calc_total_score(effective_answers, golden_ans, ScoringConfig())
sc, tot = result_score["total"], result_score["total_full"]
st.metric("选择题最终得分", f"{sc} / {tot}")
```

**关键差异**：
- `tot` 从 `len(golden_ans)` 改为 `total_full`（两者在 demo 阶段相等，但语义更准）
- 循环+集合匹配 → 一次函数调用
- 人工修正逻辑提到循环外，更清晰

### 5.2 改 `app.py:830-857`（commit 3）

在 `export_rows` 循环里加一行：

```python
row["_answers_json"] = json.dumps(r.get("answers", {}), ensure_ascii=False)
```

位置：在 `row["异常标记"] = r.get("card_flag") or ""` 后、`for q in sorted(...)` 前。

### 5.3 `golden_template.py:322-326` 不动

阶段 2 **不删** recognizer 内部的 `correct` 字段。理由：
- 阶段 1（Recognizer 抽象）才会重新设计输出结构
- 阶段 2 改 app.py 时，`correct` 字段仍可被 display 使用
- 删除会扩大改动面，增加回归风险

---

## 六、Ground Truth 策略

### 当前（demo 阶段）

- **0 份 GT**：`tests/fixtures/scoring_ground_truth.json` 文件创建但 `samples: []`
- 不写 `test_score_regression.py`（无数据可对比）

### 滚动积累（每次人工核对时）

通过 commit 3 加入的 `_answers_json` 列，**以后每次 Tab3 导出 Excel 都会自带完整答案数据**。未来某天想建立 GT 时：
1. 从最近一次人工核对过的 Excel 里挑 3-5 份
2. 填入 `tests/fixtures/scoring_ground_truth.json` 的 `samples` 数组
3. 写 `test_score_regression.py` 跑历史 baseline 对比

**这是"自然积累"而不是"前置投入"**——符合 demo 阶段"低投入验证核心逻辑"的原则。

---

## 七、实施 checklist

### Commit 1: `feat(score): 新增 ScoreCalculator 模块`

- [ ] 新建 `omr_demo/core/score_calculator.py`，80 行
- [ ] 新建 `omr_demo/tests/__init__.py`（空文件）
- [ ] 新建 `omr_demo/tests/test_score_calculator.py`，22 个 pytest 用例
- [ ] **验证 1**：`pytest omr_demo/tests/test_score_calculator.py -v` → 22 passed
- [ ] **验证 2**：`grep -E "import (streamlit|pandas)" omr_demo/core/score_calculator.py` 无输出
- [ ] **验证 3**：`git diff --stat HEAD` 只显示新增文件，无其他改动

### Commit 2: `refactor(app): Tab3 黄金模式算分改用 ScoreCalculator`

- [ ] 改 `omr_demo/app.py:810-825`，用 `calc_total_score` 替换
- [ ] 顶部 import 增加 `from core.score_calculator import calc_total_score, ScoringConfig`
- [ ] **验证 1**：启动 `streamlit run app.py`，Tab3 选 OMR0001 任意 3 张卡
- [ ] **验证 2**：每张卡的 `sc / tot` 与改造前**完全一致**（建议先用 git stash 改造前版本对比）
- [ ] **验证 3**：观察 `st.metric` 显示的分数数字精度（1.0 而非 1）

### Commit 3: `feat(export): Excel 导出含 _answers_json 列`

- [ ] 改 `omr_demo/app.py:830-857`，在 `export_rows` 加 `_answers_json` 列
- [ ] **验证 1**：导出一份 Excel
- [ ] **验证 2**：`openpyxl` 读回，找到 `_answers_json` 列，内容是合法 JSON
- [ ] **验证 3**：JSON 内容含所有题号 + answer + status

### Commit 4 (可选): `refactor(golden): 移除 recognizer 内 correct 字段`

> ⏸️ 建议前三 commit 跑稳后再决定。

---

## 八、风险与回滚

| 风险 | 触发条件 | 兜底 |
|------|---------|------|
| **多选评分规则变** | 用户改 `multi_partial=True`，但 app.py 仍按 0 分 | 保留 `multi_partial=False` 默认值；阶段 2 UI 不暴露该开关 |
| **大小写处理不一致** | `ignore_case=True` 与旧行为不符 | 跑 3 张卡的对比验证时肉眼检查；如发现差异临时改 `ignore_case=False` |
| **commit 2 分数回退** | Tab3 分数与改造前不一致 | `git revert <commit 2>`，**不动 ScoreCalculator**，先调查 app.py 调用方式 |
| **顺手优化其他代码** | 实施过程中发现"顺带可改" | **不做**，记 issue 留到阶段 1/7 一起处理 |
| **OpenCV/pandas 间接依赖** | ScoreCalculator 误 import | commit 1 验证 #2 兜底；如误 import 立即删除 |

---

## 九、验收清单（6 项硬指标）

- [ ] ① `omr_demo/core/score_calculator.py` 文件存在
- [ ] ② `pytest omr_demo/tests/test_score_calculator.py` → 22 passed
- [ ] ③ `grep -E "import (streamlit|pandas)" omr_demo/core/score_calculator.py` 无输出
- [ ] ④ `grep "set(final_ans) == set(std_ans)" omr_demo/app.py` 无输出（说明已替换）
- [ ] ⑤ 导出的 Excel 含 `_answers_json` 列，内容是合法 JSON
- [ ] ⑥ Tab3 跑 3 张测试卡，sc/tot 数字与改造前完全一致

**6 项全过 = 阶段 2 完工，进阶段 1（Recognizer 抽象）**。

---

## 十、相关文档

- [优化方案总览 §3.3 ScoreCalculator 集中式分数计算](../AI-Marker-Suite借鉴与架构优化方案.md)
- [优化方案总览 §五 阶段 1 → 2 → 7 demo 阶段路径](../AI-Marker-Suite借鉴与架构优化方案.md)
- 关联文件：
  - [omr_demo/app.py](../../omr_demo/app.py)（被改文件）
  - [omr_demo/core/golden_template.py](../../omr_demo/core/golden_template.py)（暂不改）
  - [omr_demo/core/processor.py](../../omr_demo/core/processor.py)（暂不改）
