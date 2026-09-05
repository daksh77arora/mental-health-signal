import joblib
import os
import pandas as pd
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Literal
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / 'Mental_Health_Model.pkl')
top_countries = ['Other','India','USA','Canada','Australia','UK','Germany','Mexico','Turkey','France']

app = FastAPI(title='Mental Health Signal API', version='1.0.0')
MODEL_FEATURES = [
    'Study_Hours',
    'Age',
    'Avg_Daily_Usage_Hours',
    'Daily_Unlocks',
    'Physical_Activity_Hours',
    'Sleep_Hours_Per_Night',
    'Stress_Level',
    'Gender',
    'Academic_Level',
    'Most_Used_Platform',
    'Purpose_Of_Use',
    'Grouped_country',
]

cors_origins = [origin.strip() for origin in os.getenv('CORS_ORIGINS', '').split(',') if origin.strip()]
if cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_methods=['GET', 'POST'],
        allow_headers=['Content-Type'],
    )


#A first Pydantic Model
class StudentData(BaseModel):
    age                     : int = Field(..., ge=10, le=100)
    gender                  : Literal['Male', 'Female']
    country                 : str
    academic_level          : Literal['Undergraduate', 'Graduate', 'High School']
    most_used_platform      : Literal['Facebook', 'LinkedIn', 'Instagram', 'Snapchat','Twitter','YouTube', 'TikTok', 'LINE', 'KakaoTalk', 'VKontakte', 'WhatsApp','WeChat']
    purpose_of_use          : Literal['Networking', 'Education', 'Entertainment', 'News']
    avg_daily_usage_hours   : float = Field(..., ge=0, le=24)
    daily_unlocks           : int   = Field(..., ge=0)
    study_hours             : float = Field(..., ge=0, le=24)
    physical_activity_hours : float = Field(..., ge=0, le=24)
    sleep_hours_per_night   : float = Field(..., ge=0, le=24)
    stress_level            : Literal['Medium', 'Low', 'Very High', 'High']




# Describe what we send back
class PredictionResponse(BaseModel):
    predicted_mental_health_score: float = Field(..., ge=0, le=10)
    advice: list[str]




@app.get('/', include_in_schema=False)
def greet():
    return FileResponse(BASE_DIR / 'index.html')


@app.get('/style.css', include_in_schema=False)
def stylesheet():
    return FileResponse(BASE_DIR / 'style.css', media_type='text/css')


@app.get('/script.js', include_in_schema=False)
def frontend_script():
    return FileResponse(BASE_DIR / 'script.js', media_type='application/javascript')


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.get('/model-info')
def model_info():
    return {
        'model_type': type(model).__name__,
        'score_range': [0, 10],
        'features': MODEL_FEATURES,
        'training_metrics': {
            'test_r2': 0.878,
            'test_mae': 0.347,
        },
    }


def generate_advice(data: StudentData) -> list[str]:
    advice = []

    if data.stress_level in {'High', 'Very High'}:
        advice.append('Try a short reset today: step away from the screen, breathe slowly, or talk with someone you trust.')
    if data.sleep_hours_per_night < 7:
        advice.append('Protect a consistent sleep window and reduce screen use shortly before bed.')
    if data.avg_daily_usage_hours > 6 or data.daily_unlocks > 150:
        advice.append('Create one screen-free block and silence non-essential notifications to make breaks easier.')
    if data.physical_activity_hours < 0.5:
        advice.append('Add a brief walk or stretch break; small, repeatable activity is a useful starting point.')
    if data.study_hours > 8:
        advice.append('Schedule short recovery breaks around study sessions so focused work is sustainable.')

    if not advice:
        advice.append('Keep your current rhythm steady and check in with yourself if your energy or stress changes.')

    return advice[:3]


@app.post('/predict', response_model=PredictionResponse) #6.77777
def predict(data: StudentData):
    country_group = data.country if data.country in top_countries else 'Other'

    input_row = pd.DataFrame([{
        'Age': data.age,
        'Gender': data.gender,
        'Country': data.country,
        'Academic_Level': data.academic_level,
        'Most_Used_Platform': data.most_used_platform,
        'Purpose_Of_Use': data.purpose_of_use,
        'Avg_Daily_Usage_Hours': data.avg_daily_usage_hours,
        'Daily_Unlocks': data.daily_unlocks,
        'Study_Hours': data.study_hours,
        'Physical_Activity_Hours': data.physical_activity_hours,
        'Sleep_Hours_Per_Night': data.sleep_hours_per_night,
        'Stress_Level': data.stress_level,
        'Grouped_country': country_group,
    }])

    prediction = max(0.0, min(10.0, float(model.predict(input_row)[0])))
    return PredictionResponse(
        predicted_mental_health_score=round(prediction, 2),
        advice=generate_advice(data),
    )