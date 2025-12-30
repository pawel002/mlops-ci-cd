from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    s3_bucket:                  str = "mlops-lab11-models-pjarosz"
    s3_model_dir:               str = "model"
    classifier_path:            str = "model/classifier.joblib"
    sentence_transformer_path:  str = "model/sentence_transformer.model"
    tokenizer_path:             str = "model/sentence_transformer_tokenizer"
    onnx_path:                  str = "model/classifier.onnx"
    onnx_embedding_path:        str = "model/sentence_transformer.onnx"
    embedding_dim:              int = 384