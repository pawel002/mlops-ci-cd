import pytest
import numpy as np
from unittest.mock import patch
from sentiment_app.app import Inference

class TestInference:
    
    @pytest.fixture
    def inference_instance(self, mock_settings, mock_tokenizer, mock_ort_sessions):
        embedding_session, classifier_session = mock_ort_sessions

        with patch("sentiment_app.app.Tokenizer.from_file", return_value=mock_tokenizer), \
             patch("sentiment_app.app.ort.InferenceSession", side_effect=[embedding_session, classifier_session]):
            
            inf = Inference(mock_settings)
            return inf

    def test_initialization(self, inference_instance, mock_settings):
        assert inference_instance.tokenizer is not None
        assert inference_instance.embedding_session is not None
        assert inference_instance.classifier_session is not None
        assert inference_instance.sentiment_map[0] == "negative"

    def test_predict_positive_flow(self, inference_instance, mock_tokenizer):
        inference_instance.classifier_session.run.return_value = [np.array([2])]
        
        result = inference_instance.predict("I love this code!")
        mock_tokenizer.encode.assert_called_once()
        
        embedding_call_args = inference_instance.embedding_session.run.call_args
        assert embedding_call_args[0][0] is None
        assert "input_ids" in embedding_call_args[0][1]
        assert "attention_mask" in embedding_call_args[0][1]

        classifier_call_args = inference_instance.classifier_session.run.call_args
        assert "float_input" in classifier_call_args[0][1]
        
        assert result == "positive"

    def test_predict_negative_flow(self, inference_instance):
        inference_instance.classifier_session.run.return_value = [np.array([0])]
        
        result = inference_instance.predict("This is terrible")
        assert result == "negative"

    def test_predict_unknown_label(self, inference_instance):
        inference_instance.classifier_session.run.return_value = [np.array([99])]
        
        result = inference_instance.predict("Mystery text")
        assert result == "unknown"

    def test_clean_text_integration(self, inference_instance, mock_tokenizer):
        raw_text = "  Bad \n Formatting  "
        inference_instance.predict(raw_text)
        
        called_text = mock_tokenizer.encode.call_args[0][0]
        assert called_text == "bad format"