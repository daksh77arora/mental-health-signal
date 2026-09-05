# Mental Health Signal

Student wellness analytics app that estimates a mental health score from daily habits, screen time, sleep, activity, academic level, and perceived stress. The result is informational only and is not a clinical or diagnostic assessment.

**Owner:** Daksh Arora

## Recommended GitHub Repository Name

`mental-health-signal`

Other good options are `student-wellness-signal` and `mental-health-score-app`. `mental-health-signal` matches the product name used by the frontend and avoids suggesting that the app provides a medical diagnosis.

## What Is Included

- FastAPI backend in `main.py`
- Responsive frontend in `index.html`, `style.css`, and `script.js`
- Serialized scikit-learn pipeline in `Mental_Health_Model.pkl`
- Source dataset in `Student Social Media And Mental Health Impact.csv`
- Exploratory/training notebook in `ML_Project.ipynb`

The API serves the frontend and prediction endpoint from the same origin, so the app can be deployed as one service.

## Run Locally

Use Python 3.10 or newer and install the pinned model runtime:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 2200 --reload
```

Open `http://127.0.0.1:2200/`.

- Health check: `http://127.0.0.1:2200/health`
- Model metadata: `http://127.0.0.1:2200/model-info`
- Interactive API docs: `http://127.0.0.1:2200/docs`

The model was serialized with scikit-learn `1.9.0`. Keep that version aligned with the runtime when running or deploying the app.

## API

`POST /predict` accepts JSON with these fields:

```json
{
  "age": 21,
  "gender": "Female",
  "country": "India",
  "academic_level": "Undergraduate",
  "most_used_platform": "Instagram",
  "purpose_of_use": "Education",
  "avg_daily_usage_hours": 3.5,
  "daily_unlocks": 80,
  "study_hours": 5,
  "physical_activity_hours": 1.5,
  "sleep_hours_per_night": 7.5,
  "stress_level": "Medium"
}
```

The response contains `predicted_mental_health_score`, rounded to two decimal places.

## Model Results

The bundled random-forest regression pipeline was evaluated on a held-out test split from the included dataset:

| Metric | Test result |
| --- | ---: |
| R-squared | 0.878 |
| Mean absolute error | 0.347 score points |

The model is useful as a project demonstration, but these results are not clinical validation. Performance may change on a different population or dataset.

## Tests

Run the lightweight regression and validation checks with:

```powershell
python -m unittest discover -s tests -v
```

GitHub Actions runs the same test command on every push and pull request to `main`.

## Deployment Recommendation

Deploy this first on **Render** as a Python web service. It supports FastAPI, the bundled model file, and a simple start command without requiring a separate frontend host.

- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

The service must have enough memory to load the scikit-learn pipeline. Do not commit secrets, virtual environments, caches, or private user data.

For a separate frontend origin, set `CORS_ORIGINS` to a comma-separated list of trusted origins. It is not required when the frontend is served by this API.

## Privacy and Responsible Use

This repository does not contain an owner email, login credentials, API keys, or user account records. The CSV contains row-level student attributes, so confirm its provenance and consent status before publishing it publicly. Remove or replace the dataset if it contains real, identifiable, or restricted data.

Do not use the score for diagnosis, treatment, admissions, employment, or other high-impact decisions. The UI already presents the result as informational and directs users toward trusted support.
