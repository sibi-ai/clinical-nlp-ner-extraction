"""Tests for entity-level NER evaluation metrics."""
import pytest
from src.evaluation.metrics import evaluate


def test_perfect_predictions():
    y_true = [["O", "B-CONDITION", "I-CONDITION", "O"]]
    y_pred = [["O", "B-CONDITION", "I-CONDITION", "O"]]
    results = evaluate(y_true, y_pred)
    assert results["f1"] == 1.0
    assert results["precision"] == 1.0
    assert results["recall"] == 1.0


def test_no_predictions():
    y_true = [["O", "B-MEDICATION", "O"]]
    y_pred = [["O", "O", "O"]]
    results = evaluate(y_true, y_pred)
    assert results["recall"] == 0.0


def test_partial_match():
    y_true = [["B-CONDITION", "I-CONDITION", "O", "B-MEDICATION"]]
    y_pred = [["B-CONDITION", "I-CONDITION", "O", "O"]]
    results = evaluate(y_true, y_pred)
    assert 0.0 < results["f1"] < 1.0
