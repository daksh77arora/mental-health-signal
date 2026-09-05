import unittest

from main import StudentData, generate_advice, health, model_info, predict


class PredictionTests(unittest.TestCase):
    def setUp(self):
        self.sample = StudentData(
            age=21,
            gender='Female',
            country='India',
            academic_level='Undergraduate',
            most_used_platform='Instagram',
            purpose_of_use='Education',
            avg_daily_usage_hours=3.5,
            daily_unlocks=80,
            study_hours=5,
            physical_activity_hours=1.5,
            sleep_hours_per_night=7.5,
            stress_level='Medium',
        )

    def test_health_check(self):
        self.assertEqual(health(), {'status': 'ok'})

    def test_model_info_exposes_reproducible_metadata(self):
        info = model_info()
        self.assertEqual(info['score_range'], [0, 10])
        self.assertIn('test_r2', info['training_metrics'])
        self.assertIn('Grouped_country', info['features'])

    def test_prediction_is_a_score_in_range(self):
        response = predict(self.sample)
        self.assertGreaterEqual(response.predicted_mental_health_score, 0)
        self.assertLessEqual(response.predicted_mental_health_score, 10)
        self.assertGreater(len(response.advice), 0)

    def test_advice_prioritizes_multiple_risk_inputs_without_overflowing(self):
        stressed = StudentData(**{
            **self.sample.model_dump(),
            'stress_level': 'Very High',
            'sleep_hours_per_night': 5,
            'avg_daily_usage_hours': 9,
            'daily_unlocks': 200,
            'physical_activity_hours': 0,
        })
        advice = generate_advice(stressed)
        self.assertLessEqual(len(advice), 3)
        self.assertTrue(any('sleep' in item.lower() for item in advice))

    def test_input_validation_rejects_invalid_age(self):
        with self.assertRaises(ValueError):
            StudentData(**{**self.sample.model_dump(), 'age': 9})


if __name__ == '__main__':
    unittest.main()