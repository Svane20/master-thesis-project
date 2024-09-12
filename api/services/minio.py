from typing import Dict, Any
import logging

import boto3
from fastapi import UploadFile

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Minio Configuration
MINIO_ENDPOINT = "http://127.0.0.1:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
MINIO_BUCKET = "blender-files"

# Initialize MinIO client using boto3
s3_client = boto3.client(
    's3',
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY
)


def get_minio_files(bucket_name: str = MINIO_BUCKET) -> Dict[str, Any]:
    """Get a list of files in the MinIO bucket."""
    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name)

        files = [
            {
                'key': obj['Key'],
                'lastModified': obj['LastModified'],
                'ETag': obj['ETag'],
                'size': obj['Size'],
                'storageClass': obj['StorageClass'],
                'url': f"{MINIO_ENDPOINT}/{bucket_name}/{obj['Key']}"
            }
            for obj in response.get('Contents', [])
        ]

        return {"status": "success", "files": files}
    except Exception as e:
        return {"status": "failed", "error": str(e)}


def upload_file_to_minio(file: UploadFile) -> Dict[str, Any]:
    try:
        if not bucket_exists(MINIO_BUCKET):
            s3_client.create_bucket(Bucket=MINIO_BUCKET)

        s3_client.upload_fileobj(file.file, MINIO_BUCKET, file.filename)
        file_url = f"{MINIO_ENDPOINT}/{MINIO_BUCKET}/{file.filename}"
        return {"status": "success", "file_url": file_url}
    except s3_client.exceptions.ClientError as e:
        return {"status": "failed", "error": str(e)}
    except Exception as e:
        return {"status": "failed", "error": str(e)}


def download_file_from_minio(file_name: str) -> Dict[str, Any]:
    try:
        response = s3_client.get_object(Bucket=MINIO_BUCKET, Key=file_name)
        file_stream = response['Body']

        return {"status": "success", "file_data": file_stream}

    except s3_client.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            return {"status": "failed", "error": "File not found"}
        else:
            return {"status": "failed", "error": str(e)}
    except Exception as e:
        return {"status": "failed", "error": str(e)}


def delete_file_from_minio(file_name: str) -> Dict[str, Any]:
    try:
        s3_client.delete_object(Bucket=MINIO_BUCKET, Key=file_name)
        return {"status": "success"}
    except s3_client.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            return {"status": "failed", "error": "File not found"}
        else:
            return {"status": "failed", "error": str(e)}
    except Exception as e:
        return {"status": "failed", "error": str(e)}


def bucket_exists(bucket_name: str) -> bool:
    """Check if the MinIO bucket exists."""
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        return True
    except s3_client.exceptions.NoSuchBucket:
        return False
    except Exception as e:
        logger.error(f"Error checking bucket existence: {e}")
        return False
