import boto3
from pathlib import Path
from sentiment_app.settings import Settings

def download_s3_folder(settings: Settings, destination: str = ".") -> None:
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(settings.s3_bucket)
    root_dest = Path(destination)

    for obj in bucket.objects.filter(Prefix=settings.s3_model_dir):
        if obj.key.endswith("/"):
            continue
        
        local_path = root_dest / obj.key
        local_path.parent.mkdir(parents=True, exist_ok=True)

        bucket.download_file(obj.key, str(local_path))
        print(f"Downloaded: {local_path}")