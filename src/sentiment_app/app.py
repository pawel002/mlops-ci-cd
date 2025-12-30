import os
import numpy as np
import onnxruntime as ort

from mangum import Mangum
from fastapi import FastAPI
from pydantic import BaseModel
from cleantext import clean
from tokenizers import Tokenizer

from .settings import Settings


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    prediction: str

class Inference:
    def __init__(self, settings: Settings):
        self.tokenizer = Tokenizer.from_file(os.path.join(settings.tokenizer_path, "tokenizer.json"))
        self.embedding_session = ort.InferenceSession(settings.onnx_embedding_model_path)
        self.classifier_session = ort.InferenceSession(settings.onnx_classifier_path)
        self.sentiment_map = {
            0: "negative", 
            1: "neutral", 
            2: "positive"
        }

    def predict(self, text: str) -> str:
        cleaned_text = clean(text)

        encoded = self.tokenizer.encode(cleaned_text)

        input_ids = np.array([encoded.ids])
        attention_mask = np.array([encoded.attention_mask])

        embedding_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        embeddings = self.embedding_session.run(None, embedding_inputs)[0]

        classifier_input_name = self.classifier_session.get_inputs()[0].name
        classifier_inputs = {classifier_input_name: embeddings.astype(np.float32)}
        prediction = self.classifier_session.run(None, classifier_inputs)[0]

        label = self.sentiment_map.get(prediction[0], "unknown")

        return label


app = FastAPI()
settings = Settings()
inference = Inference(settings)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest) -> PredictResponse:
    return PredictResponse(prediction=inference.predict(request.text))

handler = Mangum(app)