#type: ignore

from __future__ import annotations

import joblib


from nepalitokenizer import NepaliTokenizer

from fastapi import FastAPI
from pydantic import BaseModel


class UserInput(BaseModel):
    text: str


app = FastAPI()


model = joblib.load("model_svc.bin")


@app.get("/")
async def home() -> dict[str, str]:
    return {
        "message": "wrong address , go to /input"
    }


@app.post("/input/")
async def user_post(post: UserInput) -> dict[str, str]:
    prediction = model.predict([post.text])[0]
    return {
        "prediction": prediction
    }
