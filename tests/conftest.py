from sentiment_app.app import Settings

import pytest
import numpy as np
from unittest.mock import MagicMock

import sys

sys.modules["onnxruntime"] = MagicMock()
sys.modules["tokenizers"] = MagicMock()
sys.modules["mangum"] = MagicMock()


@pytest.fixture
def mock_settings():
    settings = MagicMock(spec=Settings)
    settings.tokenizer_path = "/fake/path"
    settings.onnx_embedding_model_path = "/fake/model.onnx"
    settings.onnx_classifier_path = "/fake/classifier.onnx"
    return settings

@pytest.fixture
def mock_tokenizer():
    tokenizer_mock = MagicMock()
    encoding_mock = MagicMock()
    encoding_mock.ids = [101, 200, 300, 102]
    encoding_mock.attention_mask = [1, 1, 1, 1]
    
    tokenizer_mock.encode.return_value = encoding_mock
    return tokenizer_mock

@pytest.fixture
def mock_ort_sessions():
    embedding_session = MagicMock()
    classifier_session = MagicMock()
    
    embedding_session.run.return_value = [np.random.rand(1, 768).astype(np.float32)]

    classifier_session.run.return_value = [np.array([2])]
    
    input_node = MagicMock()
    input_node.name = "float_input"
    classifier_session.get_inputs.return_value = [input_node]

    return embedding_session, classifier_session