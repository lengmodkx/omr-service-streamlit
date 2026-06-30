"""OMR 识别引擎包"""
from omr_service.engine.processor import CardProcessor
from omr_service.engine.standard_template import StandardTemplate
from omr_service.engine.recognizer import (
    Recognizer,
    RecognizeContext,
    RecognizeResult,
    make_recognizer,
    list_recognizer_ids,
)
from omr_service.engine.score_calculator import ScoringConfig, calc_total_score

__all__ = [
    "CardProcessor",
    "StandardTemplate",
    "Recognizer",
    "RecognizeContext",
    "RecognizeResult",
    "make_recognizer",
    "list_recognizer_ids",
    "ScoringConfig",
    "calc_total_score",
]
