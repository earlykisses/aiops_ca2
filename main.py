from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "model.joblib"

app = FastAPI(
    title="Twitter Sentiment Analysis API",
    description=(
        "Predict tweet sentiment using a trained "
        "TF-IDF and Logistic Regression model."
    ),
    version="1.0.0",
)

model = None

try:
    model = joblib.load(MODEL_PATH)
except Exception as exc:
    print(f"Model loading failed: {exc}")


class PredictionRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Tweet text to analyze",
    )

    entity: str = Field(
        default="General",
        min_length=1,
        max_length=200,
        description="Brand, product, or topic being discussed",
    )


class PredictionResponse(BaseModel):
    entity: str
    text: str
    sentiment: str
    confidence: float


@app.get("/")
def home():
    return {
        "message": "Twitter Sentiment Analysis API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded",
        )

    return {
        "status": "ok",
        "model_loaded": True,
        "model": "TF-IDF + Logistic Regression",
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded",
        )

    text = request.text.strip()
    entity = request.entity.strip()

    if not text or not entity:
        raise HTTPException(
            status_code=422,
            detail="Text and entity must not be blank",
        )

    # Match the input format used during training.
    model_input = f"entity {entity} tweet {text}"

    try:
        sentiment = model.predict([model_input])[0]

        # Logistic Regression supports predict_proba.
        probabilities = model.predict_proba([model_input])[0]
        confidence = float(max(probabilities))

        return {
            "entity": entity,
            "text": text,
            "sentiment": str(sentiment),
            "confidence": round(confidence, 4),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed",
        ) from exc