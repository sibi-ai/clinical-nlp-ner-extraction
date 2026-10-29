"""Entity-level NER evaluation metrics using seqeval."""
from seqeval.metrics import (
    classification_report, f1_score, precision_score, recall_score
)
import structlog

log = structlog.get_logger()


def evaluate(y_true: list[list[str]], y_pred: list[list[str]]) -> dict:
    """
    Compute entity-level precision, recall, F1.
    y_true / y_pred: lists of BIO tag sequences (one list per sentence).
    """
    results = {
        "precision": precision_score(y_true, y_pred),
        "recall":    recall_score(y_true, y_pred),
        "f1":        f1_score(y_true, y_pred),
        "report":    classification_report(y_true, y_pred),
    }
    log.info("eval_metrics", f1=round(results["f1"], 4),
             precision=round(results["precision"], 4),
             recall=round(results["recall"], 4))
    return results
