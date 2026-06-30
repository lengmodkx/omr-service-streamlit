"""引擎迁移回归测试"""
import os
import unittest

import cv2
import numpy as np

from omr_service.engine.recognizer import RecognizeContext, make_recognizer
from omr_service.engine.recognizers import StandardTemplateRecognizer
from omr_service.engine.standard_template import StandardTemplate


class TestEngine(unittest.TestCase):
    def test_standard_template_grid_generation(self):
        """测试列框网格生成"""
        columns = [
            {"x1": 100, "y1": 100, "x2": 200, "y2": 500, "start_q": 1, "num_q": 5, "num_options": 4}
        ]
        stp = StandardTemplate(image=None, column_configs=columns)
        self.assertEqual(len(stp.bubbles), 20)  # 5 题 * 4 选项

    def test_standard_template_answers_on_blank_image(self):
        """在空白图上识别，应返回空或不确定"""
        columns = [
            {"x1": 50, "y1": 50, "x2": 250, "y2": 250, "start_q": 1, "num_q": 2, "num_options": 4}
        ]
        img = np.full((300, 300, 3), 255, dtype=np.uint8)
        stp = StandardTemplate(image=img, column_configs=columns)
        recognizer = StandardTemplateRecognizer(standard_template=stp)
        ctx = RecognizeContext()
        result = recognizer.recognize(img, ctx)
        self.assertEqual(result.total, 2)
        self.assertTrue(all(info["status"] in ("empty", "uncertain") for info in result.answers.values()))

    def test_make_recognizer_standard(self):
        """测试工厂函数"""
        columns = [
            {"x1": 100, "y1": 100, "x2": 200, "y2": 200, "start_q": 1, "num_q": 1, "num_options": 4}
        ]
        stp = StandardTemplate(image=None, column_configs=columns)
        rec = make_recognizer("standard", standard_template=stp)
        self.assertEqual(rec.id, "standard")


if __name__ == "__main__":
    unittest.main()
