# Twitter Sentiment Analysis MLOps

An end-to-end machine learning project that predicts Twitter sentiment using a trained scikit-learn model.

## Technology Stack

- Python
- Pandas
- Scikit-learn
- TF-IDF
- Logistic Regression
- FastAPI
- Docker
- Pytest
- GitHub Actions

## Model

The trained pipeline uses TF-IDF and Logistic Regression to classify tweets into four categories:

- Positive
- Negative
- Neutral
- Irrelevant

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Application information |
| GET | `/health` | Check model availability |
| POST | `/predict` | Predict tweet sentiment |

## Example Prediction Request

```json
{
  "text": "I really love this game!",
  "entity": "Gaming"
}
```

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the API:

```bash
uvicorn main:app --reload
```

Open the interactive API documentation at:

`http://localhost:8000/docs`

## Run Tests

```bash
python -m pytest -v
```

## Run with Docker

Build the image:

```bash
docker build -t twitter-sentiment-api .
```

Start the container:

```bash
docker run -d --name twitter-sentiment-container -p 8000:8000 twitter-sentiment-api
```

Check its health:

```bash
curl http://localhost:8000/health
```

## Continuous Integration

GitHub Actions runs automated tests and builds the Docker image on pushes and pull requests to the main branch.

## Dataset

Twitter Entity Sentiment Analysis:

https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis