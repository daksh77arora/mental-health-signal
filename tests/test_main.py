import unittest

from fastapi import HTTPException

from main import StudentData, health, predict


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

    def test_prediction_is_a_score_in_range(self):
        response = predict(self.sample)
        self.assertGreaterEqual(response.predicted_mental_health_score, 0)
        self.assertLessEqual(response.predicted_mental_health_score, 10)

    def test_input_validation_rejects_invalid_age(self):
        with self.assertRaises(ValueError):
            StudentData(**{**self.sample.model_dump(), 'age': 9})


if __name__ == '__main__':
    unittest.main()