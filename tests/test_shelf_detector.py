import unittest

from src.shelf_detector import plan_fpn_levels


class PlanFpnLevelsTests(unittest.TestCase):
    def test_original_examples(self):
        boxes = [[96, 224, 130, 279], [400, 208, 417, 254], [40, 40, 340, 340]]
        for size, levels, tiny, unused in (
            (640, {2: 2, 3: 0, 4: 1, 5: 0}, 1, [3, 5]),
            (320, {2: 2, 3: 1, 4: 0, 5: 0}, 2, [4, 5]),
        ):
            with self.subTest(size=size):
                self.assertEqual(
                    plan_fpn_levels(boxes, 640, 640, size),
                    {
                        "levels": levels,
                        "tiny": tiny,
                        "unused": unused,
                        "min_input": 1280,
                    },
                )

    def test_non_square_frame_uses_separate_axis_scales(self):
        self.assertEqual(
            plan_fpn_levels([[100, 50, 548, 274]], 1280, 640, 640),
            {
                "levels": {2: 0, 3: 0, 4: 1, 5: 0},
                "tiny": 0,
                "unused": [2, 3, 5],
                "min_input": 320,
            },
        )
        self.assertEqual(
            plan_fpn_levels([[0, 0, 40, 40]], 1280, 640, 640)["tiny"], 1
        )
        self.assertEqual(
            plan_fpn_levels([[0, 0, 40, 40]], 640, 1280, 640)["tiny"], 1
        )

    def test_exact_quarter_is_acceptable(self):
        boxes = [[0, 0, 1, 1]] + [[0, 0, 64, 64]] * 3
        result = plan_fpn_levels(boxes, 640, 640, 320)
        self.assertEqual(result["tiny"], 1)
        self.assertEqual(result["min_input"], 320)

    def test_no_candidate_is_sufficient(self):
        self.assertIsNone(
            plan_fpn_levels([[0, 0, 1, 1]], 640, 640, 640)["min_input"]
        )

    def test_each_candidate_and_strict_32_pixel_boundary(self):
        for candidate in (320, 512, 640, 800, 1024, 1280):
            with self.subTest(candidate=candidate):
                result = plan_fpn_levels(
                    [[0, 0, 32, 32]], candidate, candidate, candidate
                )
                self.assertEqual(result["tiny"], 0)
                self.assertEqual(result["min_input"], candidate)

    def test_all_levels_and_clamping(self):
        boxes = [[0, 0, size, size] for size in (1, 112, 224, 448, 1024)]
        self.assertEqual(
            plan_fpn_levels(boxes, 1024, 1024, 1024)["levels"],
            {2: 1, 3: 1, 4: 1, 5: 2},
        )

    def test_empty_annotations(self):
        self.assertEqual(
            plan_fpn_levels([], 1280, 720, 640),
            {
                "levels": {2: 0, 3: 0, 4: 0, 5: 0},
                "tiny": 0,
                "unused": [2, 3, 4, 5],
                "min_input": 320,
            },
        )


if __name__ == "__main__":
    unittest.main()
