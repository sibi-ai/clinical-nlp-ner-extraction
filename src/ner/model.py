"""BERT-based NER model wrapper for clinical entity extraction."""
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import numpy as np
import structlog

log = structlog.get_logger()

LABEL2ID = {
    "O": 0,
    "B-CONDITION": 1, "I-CONDITION": 2,
    "B-MEDICATION": 3, "I-MEDICATION": 4,
    "B-PROCEDURE": 5, "I-PROCEDURE": 6,
    "B-DATE":      7, "I-DATE":      8,
    "B-DOSAGE":    9, "I-DOSAGE":   10,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


class ClinicalNERModel:
    """Inference wrapper for fine-tuned BERT NER model."""

    def __init__(self, model_path: str, device: str = "cpu"):
        self.device    = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model     = AutoModelForTokenClassification.from_pretrained(
            model_path, num_labels=len(LABEL2ID), id2label=ID2LABEL, label2id=LABEL2ID
        ).to(self.device)
        self.model.eval()
        log.info("ner_model_loaded", path=model_path, device=device)

    def predict(self, text: str) -> list[dict]:
        """Extract named entities from text. Returns list of entity dicts."""
        enc = self.tokenizer(text, return_tensors="pt",
                             truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            logits = self.model(**enc).logits[0]
        probs  = torch.softmax(logits, dim=-1).cpu().numpy()
        labels = np.argmax(probs, axis=-1)
        confs  = probs[np.arange(len(labels)), labels]

        tokens = self.tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
        return self._decode_entities(tokens, labels, confs)

    def _decode_entities(self, tokens, labels, confs) -> list[dict]:
        entities, current = [], None
        for tok, lid, conf in zip(tokens, labels, confs):
            if tok in ("[CLS]", "[SEP]", "[PAD]"):
                continue
            label = ID2LABEL[lid]
            if label.startswith("B-"):
                if current:
                    entities.append(current)
                current = {"type": label[2:], "tokens": [tok], "confidence": float(conf)}
            elif label.startswith("I-") and current and label[2:] == current["type"]:
                current["tokens"].append(tok)
                current["confidence"] = min(current["confidence"], float(conf))
            else:
                if current:
                    entities.append(current)
                current = None
        if current:
            entities.append(current)
        for ent in entities:
            ent["text"] = self.tokenizer.convert_tokens_to_string(ent.pop("tokens")).strip()
        return entities
