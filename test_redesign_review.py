import unittest

from vahan.redesign_review import build_review_text


class RedesignReviewTests(unittest.TestCase):
    def test_review_keeps_measured_trade_visible(self):
        text = build_review_text(
            car={'rack_length_mm': 591.5, 'front_bump_steer_limit_deg': 0.15},
            steer={'total_rack_travel_mm': 97, 'rack_travel_per_rev_mm': 102.71,
                   'rack_direction': -1},
            kinematic={'bump_steer_span_deg': 0.147, 'ackermann_pct': 54.615},
            alignment={'front_camber_deg': -0.776, 'rear_camber_deg': -0.7},
        )
        self.assertIn('0.147°', text)
        self.assertIn('former 0.005° target', text)
        self.assertIn('Class A and Class B', text)
        self.assertIn('−X', text)

    def test_unknown_derived_values_are_explicit(self):
        text = build_review_text()
        self.assertIn('n/a', text)
        self.assertIn('not run', text)


if __name__ == '__main__':
    unittest.main()
