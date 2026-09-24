from unittest.mock import MagicMock
import numpy as np
import pytest

from omr_service.core.service import OmrService
from omr_service.core.exceptions import (
    TemplateNotFoundError,
    ImageLoadError,
    InternalError,
)


@pytest.fixture
def mock_deps():
    return {
        "template_store": MagicMock(),
        "image_loader": MagicMock(),
        "worker_pool": MagicMock(),
        "ocr_engine": MagicMock(),
        "cropper": MagicMock(),
    }


@pytest.fixture
def service(mock_deps):
    return OmrService(**mock_deps)


def test_recognize_returns_code_0_on_success(service, mock_deps):
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_deps["image_loader"].load_multi.return_value = [img]
    mock_deps["template_store"].get.return_value = MagicMock()
    mock_deps["worker_pool"].submit.return_value.result.return_value = (
        [
            {"question_no": 1, "selected": ["A"], "is_blank": False, "is_multiple": False, "answer_type": "single"}
        ],
        False,
    )

    result = service.recognize({
        "template_id": 1,
        "scan_image_urls": ["http://x/y.jpg"],
    })

    assert result["code"] == 0
    assert result["template_id"] == 1
    assert "answers" in result
    # 与 Java 端 OmrResult 对齐的 primitive 字段必须存在
    assert "abnormal" in result
    assert "empty_count" in result
    assert "multiple_count" in result
    for a in result["answers"]:
        assert "is_blank" in a
        assert "is_multiple" in a
    assert "elapsed_ms" in result


def test_recognize_raises_template_not_found(service, mock_deps):
    mock_deps["image_loader"].load_multi.return_value = [np.zeros((10, 10, 3), dtype=np.uint8)]
    mock_deps["template_store"].get.return_value = None

    with pytest.raises(TemplateNotFoundError):
        service.recognize({
            "template_id": "missing",
            "scan_image_urls": ["http://x/y.jpg"],
        })


def test_recognize_raises_image_load_error(service, mock_deps):
    mock_deps["image_loader"].load_multi.side_effect = FileNotFoundError("404")

    with pytest.raises(ImageLoadError):
        service.recognize({
            "template_id": 1,
            "scan_image_urls": ["http://x/bad.jpg"],
        })


def test_parse_golden_template_returns_code_0(service, mock_deps):
    mock_deps["image_loader"].load.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

    result = service.parse_golden_template({
        "template_id": 1,
        "template_image_url": "http://x/tpl.jpg",
        "columns": [
            {
                "x1": 0, "y1": 0, "x2": 100, "y2": 200,
                "start_q": 1, "num_q": 5, "num_options": 4,
            }
        ],
    })

    assert result["code"] == 0
    assert result["template_id"] == 1
    assert "answers" in result


def test_parse_golden_template_empty_columns(service, mock_deps):
    """columns 为空时返回 code 0 + 空答案（多页模板中无选择题列的页合法）"""
    mock_deps["image_loader"].load.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

    result = service.parse_golden_template({
        "template_id": 1,
        "template_image_url": "http://x/tpl.jpg",
        "columns": [],
    })

    assert result["code"] == 0
    assert result["answers"] == []
    assert result["bubble_grid"] == []


def test_verify_recognition_rate_not_implemented(service):
    with pytest.raises(InternalError):
        service.verify_recognition_rate({})


def test_reverify_paper_delegates_to_recognize(service, mock_deps):
    mock_deps["image_loader"].load_multi.return_value = [np.zeros((10, 10, 3), dtype=np.uint8)]
    mock_deps["template_store"].get.return_value = MagicMock()
    mock_deps["worker_pool"].submit.return_value.result.return_value = ([], False)

    result = service.reverify_paper({
        "template_id": 1,
        "scan_image_urls": ["http://x/y.jpg"],
    })

    assert result["code"] == 0


def test_recognize_includes_ocr_personal_info(service, mock_deps):
    """OCR 识别个人信息 (mock 返回)."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_deps["image_loader"].load_multi.return_value = [img]
    mock_deps["template_store"].get.return_value = MagicMock()
    mock_deps["worker_pool"].submit.return_value.result.return_value = ([], False)
    # _recognize_personal_info 期望 ocr_engine.recognize 返回 list[dict]（与区域一一对应）
    mock_deps["ocr_engine"].recognize.return_value = [
        {"field": "name", "value": "张三", "confidence": 0.95}
    ]

    result = service.recognize({
        "template_id": 1,
        "scan_image_urls": ["http://x/y.jpg"],
        "personal_info_region": {"field": "name", "x1": 0, "y1": 0, "x2": 50, "y2": 50},
    })

    assert "personal_info" in result
    assert result["personal_info"][0]["value"] == "张三"


def test_recognize_includes_subjective_crops(service, mock_deps):
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_deps["image_loader"].load_multi.return_value = [img]
    mock_deps["template_store"].get.return_value = MagicMock()
    mock_deps["worker_pool"].submit.return_value.result.return_value = ([], False)
    # 实现调用的是 crop_subjective_regions（不是 crop）
    mock_deps["cropper"].crop_subjective_regions.return_value = [
        {"q": 1, "image_url": "http://x/c1.jpg", "page_index": 0}
    ]

    result = service.recognize({
        "template_id": 1,
        "scan_image_urls": ["http://x/y.jpg"],
        "subjective_regions": [{"q": 1, "x1": 0, "y1": 0, "x2": 100, "y2": 50}],
    })

    assert "subjective_crops" in result
    assert result["subjective_crops"][0]["image_url"] == "http://x/c1.jpg"


def test_parse_golden_template_with_personal_info(service, mock_deps):
    mock_deps["image_loader"].load.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_deps["ocr_engine"].recognize.return_value = [
        {"field": "name", "value": "张三", "confidence": 0.95}
    ]

    result = service.parse_golden_template({
        "template_id": 1,
        "template_image_url": "http://x/tpl.jpg",
        "columns": [
            {
                "x1": 0, "y1": 0, "x2": 100, "y2": 200,
                "start_q": 1, "num_q": 5, "num_options": 4,
            }
        ],
        "personal_info_region": [
            {"field": "name", "x1": 0, "y1": 0, "x2": 50, "y2": 50},
        ],
    })

    assert "personal_info_sample" in result
    assert result["personal_info_sample"][0]["value"] == "张三"


def test_run_with_timeout_returns_value():
    from omr_service.core.service import run_with_timeout

    def ok():
        return "done"

    assert run_with_timeout(ok, 2.0, "测试", "default") == "done"


def test_run_with_timeout_timeout_returns_default():
    import time
    from omr_service.core.service import run_with_timeout

    def slow():
        time.sleep(10)

    t0 = time.monotonic()
    assert run_with_timeout(slow, 0.3, "测试", "default") == "default"
    assert time.monotonic() - t0 < 5  # 不应阻塞到函数真正结束


def test_run_with_timeout_raises_inner_exception():
    from omr_service.core.service import run_with_timeout

    def boom():
        raise ValueError("inner boom")

    with pytest.raises(ValueError):
        run_with_timeout(boom, 2.0, "测试", "default")


# ===== FastAPI 重写回归修复的行为测试 (2026-08-04) =====

from omr_service.core.service import (
    normalize_columns,
    normalize_personal_info,
    normalize_subjective_regions,
)


def test_normalize_columns_snake_case_passthrough():
    cols = normalize_columns([
        {"x1": 1, "y1": 2, "x2": 100, "y2": 200, "start_q": 3, "num_q": 5, "num_options": 4}
    ])
    assert cols == [{
        "x1": 1, "y1": 2, "x2": 100, "y2": 200,
        "start_q": 3, "num_q": 5, "num_options": 4,
        "option_axis": "x", "reverse_q": False, "page_index": 0,
    }]


def test_normalize_columns_java_sync_naming():
    """Java 同步 HTTP 链路 (buildFastApiColumns) 的 question_start/question_count/options_per_question"""
    cols = normalize_columns([
        {"x1": 1, "y1": 2, "x2": 100, "y2": 200,
         "question_start": 1, "question_count": 5, "options_per_question": 3,
         "option_axis": "y", "reverse_q": True, "page_index": 1}
    ])
    assert cols[0]["start_q"] == 1
    assert cols[0]["num_q"] == 5
    assert cols[0]["num_options"] == 3
    assert cols[0]["option_axis"] == "y"
    assert cols[0]["reverse_q"] is True
    assert cols[0]["page_index"] == 1


def test_normalize_columns_camelcase_and_xywh():
    """MQ 原始 camelCase + x/y/width/height 坐标写法"""
    cols = normalize_columns([
        {"x": 10, "y": 20, "width": 90, "height": 180,
         "startQ": 1, "numQ": 5, "numOptions": 4, "pageIndex": 0}
    ])
    assert cols[0]["x2"] == 100
    assert cols[0]["y2"] == 200
    assert cols[0]["start_q"] == 1
    assert cols[0]["num_q"] == 5


def test_normalize_columns_invalid_skipped():
    """缺坐标或 num_q<=0 的列跳过而不是抛 KeyError"""
    cols = normalize_columns([
        {"start_q": 1, "num_q": 5},  # 无坐标
        {"x1": 0, "y1": 0, "x2": 1, "y2": 1, "question_start": 1, "question_count": 0},  # num_q=0
    ])
    assert cols == []


def test_normalize_personal_info_camelcase():
    regions = normalize_personal_info([{"field": "name", "x1": 0, "y1": 0, "x2": 5, "y2": 5, "pageIndex": 1}])
    assert regions[0]["page_index"] == 1


def test_normalize_subjective_regions_camelcase():
    regions = normalize_subjective_regions([{"q": 53, "x1": 0, "y1": 0, "x2": 5, "y2": 5, "pageIndex": 1, "stitchWithNext": True}])
    assert regions[0]["page_index"] == 1
    assert regions[0]["stitch_with_next"] is True


def test_crop_subjective_single_image_page_normalized_and_restored(service, mock_deps):
    """单图场景（Java 按页拆请求）：page_index=1 的区域应归一化为 0 选图，结果还原原始页码"""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_deps["cropper"].crop_subjective_regions.return_value = [
        {"q": 53, "image_url": "http://x/q53.jpg", "page_index": 0}
    ]

    crops = service._crop_subjective(
        [img],
        [{"q": 53, "x1": 0, "y1": 0, "x2": 50, "y2": 50, "page_index": 1}],
        "ns",
    )

    # 裁剪器收到的 page_index 应被归一化为 0
    called_regions = mock_deps["cropper"].crop_subjective_regions.call_args[0][1]
    assert called_regions[0]["page_index"] == 0
    # 返回结果应还原原始页码 1
    assert crops[0]["page_index"] == 1


def test_recognize_personal_info_single_image_page_normalized(service, mock_deps):
    """单图场景下 page_index=1 的个人信息区域不应越界丢空"""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_deps["ocr_engine"].recognize.return_value = [
        {"field": "name", "value": "张三", "confidence": 0.95}
    ]

    results = service._recognize_personal_info(
        [img], [{"field": "name", "x1": 0, "y1": 0, "x2": 5, "y2": 5, "page_index": 1}]
    )
    assert results[0]["value"] == "张三"


def test_recognize_personal_info_block_fields_flattened(service, mock_deps):
    """考生信息区解析出的子字段应平铺追加（Java 端按 field 平铺读取）"""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_deps["ocr_engine"].recognize_block.return_value = {"raw_text": "姓名:张三 考场:1"}

    import omr_service.core.service as svc_mod
    from omr_service.engine import personal_info_block_parser

    orig = personal_info_block_parser.parse_personal_info_block
    personal_info_block_parser.parse_personal_info_block = lambda t, **kwargs: ({"name": "张三", "room": "1"}, 0.9)
    # service 内部 from import，需要打补丁到引擎模块
    try:
        results = service._recognize_personal_info(
            [img], [{"field": "student_info_block", "x1": 0, "y1": 0, "x2": 5, "y2": 5, "page_index": 0}]
        )
    finally:
        personal_info_block_parser.parse_personal_info_block = orig

    fields = {r["field"]: r["value"] for r in results}
    # 块原文末尾会回写最终姓名的“印刷体姓名”标记行（供 Java 端二次解析对齐）
    assert fields["student_info_block"].startswith("姓名:张三 考场:1")
    assert "印刷体姓名:张三" in fields["student_info_block"]
    assert fields["name"] == "张三"
    assert fields["room"] == "1"


def test_recognize_personal_info_low_confidence_cleared(service, mock_deps):
    """低于 ocr_confidence_threshold 的结果 value 置空"""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_deps["ocr_engine"].recognize.return_value = [
        {"field": "name", "value": "张?", "confidence": 0.1}
    ]

    results = service._recognize_personal_info(
        [img], [{"field": "name", "x1": 0, "y1": 0, "x2": 5, "y2": 5, "page_index": 0}]
    )
    assert results[0]["value"] == ""


def test_parse_golden_template_caches_real_page_index(service, mock_deps):
    """多页解析时模板缓存 page_images 应使用真实页码作为键，避免后页覆盖前页"""
    mock_deps["image_loader"].load.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

    service.parse_golden_template({
        "template_id": 1,
        "template_image_url": "http://x/page2.jpg",
        "columns": [],
        "page_index": 1,
        "subjective_regions": [{"q": 53, "x1": 0, "y1": 0, "x2": 50, "y2": 50, "pageIndex": 1}],
    })

    cached = mock_deps["template_store"].set.call_args[0][1]
    assert list(cached.page_images.keys()) == [1]
    # 主观题区域 camelCase 已归一化
    assert cached.subjective_regions[0]["page_index"] == 1
    assert cached.subjective_regions[0]["stitch_with_next"] is False
