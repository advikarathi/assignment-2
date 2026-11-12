"""FastAPI app serving the scikit-learn term-deposit pipeline.

Run with:  uv run fastapi dev main.py   then open http://127.0.0.1:8000/docs
"""

from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Loaded once, when the app starts - not on every request.
model = joblib.load("model.joblib")

app = FastAPI(
    title="Term Deposit Predictor",
    description="Predicts whether a bank client will subscribe to a term deposit "
                "if called in the next campaign.",
)


class Client(BaseModel):
    age: int = Field(ge=18, le=100, examples=[35])
    job: Literal["admin.", "blue-collar", "entrepreneur", "housemaid", "management",
                 "retired", "self-employed", "services", "student", "technician",
                 "unemployed", "unknown"] = Field(examples=["management"])
    marital: Literal["married", "single", "divorced"] = Field(examples=["married"])
    education: Literal["primary", "secondary", "tertiary", "unknown"] = Field(examples=["tertiary"])
    default: Literal["yes", "no"] = Field(description="Has credit in default?", examples=["no"])
    balance: int = Field(description="Average yearly balance (euros)", examples=[1500])
    housing: Literal["yes", "no"] = Field(description="Has a housing loan?", examples=["yes"])
    loan: Literal["yes", "no"] = Field(description="Has a personal loan?", examples=["no"])
    contact: Literal["cellular", "telephone", "unknown"] = Field(examples=["cellular"])
    day: int = Field(ge=1, le=31, description="Day of month of the planned contact", examples=[15])
    month: Literal["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug",
                   "sep", "oct", "nov", "dec"] = Field(examples=["may"])
    campaign: int = Field(ge=1, description="Contacts in this campaign, including this one", examples=[1])
    pdays: int = Field(description="Days since last contact in a previous campaign (-1 = never)",
                       examples=[-1])
    previous: int = Field(ge=0, description="Contacts before this campaign", examples=[0])
    poutcome: Literal["success", "failure", "other", "unknown"] = Field(
        description="Outcome of the previous campaign", examples=["unknown"])


class Prediction(BaseModel):
    subscribe: Literal["yes", "no"]


@app.post("/predict", response_model=Prediction)
def predict(client: Client) -> Prediction:
    row = pd.DataFrame([client.model_dump()])
    return Prediction(subscribe=model.predict(row)[0])
