from datetime import datetime, timedelta, date
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="Cattle Heat Detection & Breeding Prediction API")

# Allow the web app (Lovable) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("heat_model.pkl")

FEATURES = ["activity_level", "mounting", "mucus_discharge", "restlessness",
            "milk_drop_pct", "temperature", "feed_intake_pct"]

# Breeding constants for cattle
CYCLE_DAYS = 21          # estrus cycle
AI_HOURS_AFTER_ONSET = 12  # AM/PM rule
GESTATION_DAYS = 283
PREG_CHECK_DAYS = 35     # pregnancy check 30-60 days after AI


class Observation(BaseModel):
    activity_level: float = Field(..., ge=1, le=10)
    mounting: bool
    mucus_discharge: bool
    restlessness: float = Field(..., ge=1, le=10)
    milk_drop_pct: float = Field(..., ge=0, le=100)
    temperature: float = Field(..., ge=35, le=42)
    feed_intake_pct: float = Field(..., ge=0, le=120)
    obs_time: Optional[datetime] = None


class Breeding(BaseModel):
    ai_date: date


@app.get("/")
def home():
    return {"status": "running", "docs": "/docs"}


@app.post("/predict-heat")
def predict_heat(obs: Observation):
    row = pd.DataFrame([{
        "activity_level": obs.activity_level,
        "mounting": int(obs.mounting),
        "mucus_discharge": int(obs.mucus_discharge),
        "restlessness": obs.restlessness,
        "milk_drop_pct": obs.milk_drop_pct,
        "temperature": obs.temperature,
        "feed_intake_pct": obs.feed_intake_pct,
    }])[FEATURES]

    prob = float(model.predict_proba(row)[0][1])
    level = "High" if prob >= 0.70 else "Medium" if prob >= 0.40 else "Low"

    when = obs.obs_time or datetime.utcnow()
    in_heat = prob >= 0.40

    return {
        "heat_probability": round(prob, 3),
        "heat_level": level,
        "best_ai_time": (when + timedelta(hours=AI_HOURS_AFTER_ONSET)).isoformat() if in_heat else None,
        "next_heat_date": (when.date() + timedelta(days=CYCLE_DAYS)).isoformat() if in_heat else None,
        "advice": {
            "High": "Strong signs of heat. Inseminate about 12 hours from now.",
            "Medium": "Possible heat. Observe again in 4-6 hours.",
            "Low": "No clear signs of heat. Continue routine monitoring.",
        }[level],
    }


@app.post("/breeding-plan")
def breeding_plan(b: Breeding):
    return {
        "ai_date": b.ai_date.isoformat(),
        "return_to_heat_check": (b.ai_date + timedelta(days=CYCLE_DAYS)).isoformat(),
        "pregnancy_check_date": (b.ai_date + timedelta(days=PREG_CHECK_DAYS)).isoformat(),
        "expected_calving": (b.ai_date + timedelta(days=GESTATION_DAYS)).isoformat(),
    }
