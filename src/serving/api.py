"""FastAPI NER serving endpoint."""
from fastapi import FastAPI
from pydantic import BaseModel
import structlog

app = FastAPI(title="Clinical NER API")
log = structlog.get_logger()
_model = None


def get_model():
    global _model
    if _model is None:
        import os
        from src.ner.model import ClinicalNERModel
        _model = ClinicalNERModel(os.environ.get("MODEL_PATH", "models/clinical-ner"))
    return _model


class NERRequest(BaseModel):
    text: str


class Entity(BaseModel):
    type: str
    text: str
    confidence: float


class NERResponse(BaseModel):
    entities: list[Entity]
    entity_count: int


@app.post("/extract", response_model=NERResponse)
async def extract_entities(req: NERRequest):
    model  = get_model()
    entities = model.predict(req.text)
    log.info("ner_prediction", n_entities=len(entities))
    return NERResponse(
        entities=[Entity(**e) for e in entities],
        entity_count=len(entities),
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
