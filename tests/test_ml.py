import pytest
from src.ml.dataset_prep import generate_sample_dataset, CATEGORIES
from src.ml.predictor import DomainClassifierPredictor

def test_dataset_generation():
    texts, labels = generate_sample_dataset()
    assert len(texts) > 0
    assert len(labels) == len(texts)
    assert max(labels) < len(CATEGORIES)

def test_domain_predictor_inference():
    predictor = DomainClassifierPredictor()
    sample_text = "Kubernetes microservices deployment on Docker containers with auto-scaling pods."
    
    result = predictor.predict_category(sample_text)
    assert "category" in result
    assert "confidence" in result
    assert "probabilities" in result
    assert result["category"] in CATEGORIES
