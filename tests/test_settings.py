from sentiment_app.settings import Settings


def test_settings_correct():
    settings = Settings()
    assert settings.s3_bucket == "mlops-lab11-models-pjarosz"
    assert settings.s3_model_dir == "model"
    assert settings.classifier_joblib_path == "model/classifier.joblib"
    assert settings.sentence_transformer_dir == "model/sentence_transformer.model"
    assert settings.tokenizer_path == "model/sentence_transformer_tokenizer"
    assert settings.onnx_classifier_path == "model/classifier.onnx"
    assert settings.onnx_embedding_model_path == "model/sentence_transformer.onnx"
    assert settings.embedding_dim == 384