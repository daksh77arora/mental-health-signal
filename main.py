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
    return PredictionResponse(predicted_mental_health_score=round(prediction, 2))