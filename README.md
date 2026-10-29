# Clinical NLP — NER & Information Extraction

BERT-based named entity recognition and information extraction pipeline
for unstructured clinical and medical text.

## Features

- Fine-tuned BERT / BioBERT for clinical NER (conditions, medications, procedures, dates)
- Relation extraction between extracted entities
- LLM-based structured extraction with JSON schema validation
- Confidence scoring and uncertainty estimation on predictions
- FastAPI serving with async inference
- Evaluation harness with entity-level F1, precision, recall

## Entity Types

| Type | Examples |
|------|---------|
| CONDITION | hypertension, type 2 diabetes |
| MEDICATION | metformin 500mg, lisinopril |
| PROCEDURE | chest X-ray, MRI, blood panel |
| DATE | 12/03/2024, last week |
| DOSAGE | 500mg twice daily |

## Stack

| Component | Technology |
|-----------|-----------|
| NER model | BERT / BioBERT (HuggingFace) |
| Training | PyTorch, HuggingFace Trainer |
| Structured extraction | OpenAI GPT-4o with JSON schema |
| Serving | FastAPI |
| Evaluation | seqeval (entity-level F1) |

## Project Structure

```
├── src/
│   ├── ner/            # BERT NER model, training, inference
│   ├── extraction/     # LLM-based structured extraction
│   ├── evaluation/     # Entity-level metrics
│   └── serving/        # FastAPI endpoints
├── data/               # Training data format examples
├── notebooks/          # EDA, error analysis
├── configs/
├── tests/
├── Dockerfile
└── requirements.txt
```

## Quickstart

```bash
git clone https://github.com/sibi-ai/clinical-nlp-ner-extraction
cd clinical-nlp-ner-extraction
pip install -r requirements.txt
python src/ner/train.py --config configs/ner.yaml
uvicorn src.serving.api:app --reload
```
