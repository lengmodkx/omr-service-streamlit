from pydantic import BaseModel, Field

from omr_service.api.schemas.common import BaseResponse, SubjectiveRegion


class QuestionAnswer(BaseModel):
    """单题答案结构（模板解析等场景使用）"""
    question_no: int
    answer: str = ""
    status: str = "empty"
    correct: bool | None = None


class AnswerInfo(BaseModel):
    """识别结果中单题答案（list 元素）"""
    question_no: int
    selected: list[str] = Field(default_factory=list)
    status: str = "empty"
    is_blank: bool = False
    is_multiple: bool = False
    correct: bool | None = None


class BubblePoint(BaseModel):
    """气泡坐标（前端可视化用）"""
    q: int
    opt: str
    x: int
    y: int
    w: int
    h: int


class PersonalInfoEntry(BaseModel):
    """单条个人信息字段（personal_info list 元素）"""
    field: str
    value: str = ""
    confidence: float = 0.0
    parsed_fields: dict | None = None


class RecognizeRequest(BaseModel):
    """同步识别请求"""
    template_id: str | int  # 兼容 Java 端发字符串
    scan_image_urls: list[str] = Field(min_length=1)
    question_no: int | None = None
    personal_info_region: list | dict | None = None  # 接受 list[region] 或单个 dict
    subjective_regions: list[SubjectiveRegion] | None = None
    # 考生名单（花名册），可选；提供后姓名识别结果按名单相似度裁决纠正
    candidate_names: list[str] | None = None


class SubjectiveCrop(BaseModel):
    """主观题裁剪结果（与 SubjectiveCropper.crop_subjective_regions 输出对齐）"""
    q: int = 0
    image_url: str = ""
    page_index: int = 0


class RecognizeResponse(BaseResponse):
    """同步识别响应（与 OmrService.recognize 实际返回结构对齐）"""
    template_id: int
    answers: list[AnswerInfo] = Field(default_factory=list)
    bubbles: list[BubblePoint] = Field(default_factory=list)
    bubble_grid: list[BubblePoint] = Field(default_factory=list)
    empty_count: int = 0
    multiple_count: int = 0
    abnormal: bool = False
    personal_info: list[PersonalInfoEntry] = Field(default_factory=list)
    subjective_crops: list[SubjectiveCrop] = Field(default_factory=list)
    elapsed_ms: int = 0
