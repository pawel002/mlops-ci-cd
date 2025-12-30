import sys
from unittest.mock import MagicMock
import pytest
import numpy as np

# Mock tokenizers
global_tokenizer_instance = MagicMock()
global_tokenizer_instance.encode.return_value.ids = [101, 200, 300, 102]
global_tokenizer_instance.encode.return_value.attention_mask = [1, 1, 1, 1]

mock_tokenizers_lib = MagicMock()
mock_tokenizers_lib.Tokenizer.from_file.return_value = global_tokenizer_instance
sys.modules["tokenizers"] = mock_tokenizers_lib

# mock onnx runtime
global_embedding_session = MagicMock()
global_classifier_session = MagicMock()

global_embedding_session.run.return_value = [np.random.rand(1, 384).astype(np.float32)]
global_classifier_session.run.return_value = [np.array([2])]
input_node = MagicMock()
input_node.name = "float_input"
global_classifier_session.get_inputs.return_value = [input_node]

mock_ort_lib = MagicMock()
def side_effect_session(path):
    if "sentence_transformer" in path:
        return global_embedding_session
    return global_classifier_session

mock_ort_lib.InferenceSession.side_effect = side_effect_session
sys.modules["onnxruntime"] = mock_ort_lib

from sentiment_app.app import Settings

@pytest.fixture
def mock_settings():
    return Settings(
        s3_bucket="mock-bucket",
        s3_model_dir="mock-dir",
        classifier_joblib_path="mock/classifier.joblib",
        sentence_transformer_dir="mock/sentence_transformer.model",
        tokenizer_path="mock/tokenizer",
        onnx_classifier_path="mock/classifier.onnx",
        onnx_embedding_model_path="mock/embedding.onnx",
        embedding_dim=384
    )

@pytest.fixture
def mock_tokenizer():
    global_tokenizer_instance.reset_mock()
    return global_tokenizer_instance

@pytest.fixture
def mock_ort_sessions():
    global_embedding_session.reset_mock()
    global_classifier_session.reset_mock()
    return global_embedding_session, global_classifier_session