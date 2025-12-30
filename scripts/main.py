import typer
import logging
from typing_extensions import Annotated
from download_artifacts import download_s3_folder
from export_classifier_to_onnx import export_classifier_to_onnx
from export_sentence_transformer_to_onnx import export_model_to_onnx
from sentiment_app.settings import Settings

app = typer.Typer()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.command()
def main(
    download_artifacts: Annotated[bool, typer.Option("--download", help="Download artifacts from S3")] = False,
    export_models: Annotated[bool, typer.Option("--export", help="Export models to ONNX")] = False,
):
    if not (download_artifacts or export_models):
        typer.echo("Please provide a flag: --download or --export")
        raise typer.Exit(code=1)

    settings = Settings()

    if download_artifacts:
        logger.info("Downloading artifacts...")
        download_s3_folder(settings)

    if export_models:
        logger.info("Exporting models...")
        export_classifier_to_onnx(settings)
        export_model_to_onnx(settings)

if __name__ == "__main__":
    app()