from pydantic import BaseModel, ConfigDict, Field

from omr_service.api.schemas.common import BaseResponse, SubjectiveRegion


class ColumnConfig(BaseModel):
    """选择题单列配置（黄金模板解析入参）

    字段命名同时支持 snake_case 与 camelCase（Java 端 buildFastApiColumns 使用 camelCase）。
    StandardTemplate 实际需要 x1/y1/x2/y2 + start_q/num_q/num_options 三个核心坐标/题号字段。
    """

    model_config = ConfigDict(populate_by_name=True)

    column_id: str | None = Field(default=None, alias="columnId")
    column_index: int = 0
    question_start: int | None = Field(default=None, alias="questionStart")
    question_count: int | None = Field(default=None, alias="questionCount")
    question_type: str | None = Field(default=None, alias="questionType")
    options_per_question: int | None = Field(default=4, alias="optionsPerQuestion")

    # 兼容字段:直接传 x1/y1/x2/y2
    x1: int | None = None
    y1: int | None = None
    x2: int | None = None
    y2: int | None = None

    # 兼容字段:穿 x/y/width/height (Java 端 buildColumns 风格)
    x: int | None = None
    y: int | None = None
    width: int | None = None
    height: int | None = None
    page_index: int | None = Field(default=None, alias="pageIndex")


class GoldenTemplateRequest(BaseModel):
    """黄金模板解析请求"""
    model_config = ConfigDict(populate_by_name=True)

    template_id: str | int
    template_image_url: str
    # 允许空列表：多页模板中只有个人信息/主观题的页没有选择题列
    columns: list[ColumnConfig] = Field(default_factory=list)
    personal_info_region: list | None = Field(default=None, alias="personalInfoRegion")
    subjective_regions: list[SubjectiveRegion] | None = Field(default=None, alias="subjectiveRegions")


class AnswerEntry(BaseModel):
    """单题答案（解析后）

    is_blank/is_multiple 与 Java 端 OmrResult.QuestionAnswer 的 primitive 字段对齐。
    """
    question_no: int
    selected: list[str] = Field(default_factory=list)
    status: str = "empty"
    correct: bool | None = None
    is_blank: bool = False
    is_multiple: bool = False


class BubbleGrid(BaseModel):
    """气泡坐标（前端预览）

    service 返回的键为 snake_case 的 page_index，需开启 populate_by_name 才能按字段名
    取值（否则会因 alias="pageIndex" 而回退默认值 0，多页气泡页码全部丢失）。
    """
    model_config = ConfigDict(populate_by_name=True)

    q: int
    opt: str
    x: int
    y: int
    w: int
    h: int
    page_index: int | None = Field(default=0, alias="pageIndex")


class PersonalInfoEntry(BaseModel):
    field: str
    value: str = ""
    confidence: float = 0.0


class GoldenTemplateResponse(BaseResponse):
    """黄金模板解析响应（与 OmrService.parse_golden_template 实际返回对齐）"""
    template_id: int
    answers: list[AnswerEntry] = Field(default_factory=list)
    bubble_grid: list[BubbleGrid] = Field(default_factory=list)
    personal_info_sample: list[PersonalInfoEntry] = Field(default_factory=list)
    subjective_crops: list[dict] = Field(default_factory=list)
    # Java 端 OmrResult 的 primitive 字段，必须始终出现在响应中
    abnormal: bool = False
    empty_count: int = 0
    multiple_count: int = 0
    elapsed_ms: int = 0
