from typing import Dict, Any
import logging

import boto3
from fastapi import UploadFile

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Minio Configuration
# MINIO_ENDPOINT = "http://127.0.0.1:9000"
MINIO_ENDPOINT = "http://minio.minio:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
MINIO_BUCKET_BLENDER = "blender-files"
MINIO_BUCKET_IMAGES = "rendered-images"

# Initialize MinIO client using boto3
s3_client = boto3.client(
    's3',
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY
)


def get_minio_files(bucket_name: str = MINIO_BUCKET_BLENDER) -> Dict[str, Any]:
    """Get a list of files in the MinIO bucket."""
    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name)
        if 'Contents' not in response:
            return {"status": "success", "files": []}

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
        return handle_minio_error(e)


def upload_file_to_minio(file: UploadFile, bucket_name: str = MINIO_BUCKET_BLENDER) -> Dict[str, Any]:
    try:
        # Ensure the bucket exists or create it
        if not bucket_exists(bucket_name):
            logger.info(f"Bucket {bucket_name} does not exist. Creating bucket.")
            s3_client.create_bucket(Bucket=bucket_name)

        # Upload file
        s3_client.upload_fileobj(file.file, bucket_name, file.filename)
        file_url = f"{MINIO_ENDPOINT}/{bucket_name}/{file.filename}"
        logger.info(f"File {file.filename} uploaded successfully to {file_url}.")
        return {"status": "success", "file_url": file_url}
    except Exception as e:
        return handle_minio_error(e)


def upload_file_to_minio_from_path(file_path: str, key: str, bucket_name: str = MINIO_BUCKET_IMAGES) -> Dict[str, Any]:
    try:
        # Ensure the bucket exists or create it
        if not bucket_exists(bucket_name):
            logger.info(f"Bucket {bucket_name} does not exist. Creating bucket.")
            s3_client.create_bucket(Bucket=bucket_name)

        # Upload file
        with open(file_path, 'rb') as file_data:
            s3_client.upload_fileobj(file_data, bucket_name, key)

        file_url = f"{MINIO_ENDPOINT}/{bucket_name}/{key}"
        logger.info(f"File {key} uploaded successfully to {file_url}.")
        return {"status": "success", "file_url": file_url}
    except Exception as e:
        logger.error(f"Failed to upload {file_path} to MinIO: {e}")
        return handle_minio_error(e)


def download_file_from_minio(file_name: str, bucket_name: str = MINIO_BUCKET_BLENDER) -> Dict[str, Any]:
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=file_name)
        file_stream = response['Body']
        logger.info(f"File {file_name} downloaded successfully.")
        return {"status": "success", "file_data": file_stream}
    except Exception as e:
        return handle_minio_error(e)


def delete_file_from_minio(file_name: str, bucket_name: str = MINIO_BUCKET_BLENDER) -> Dict[str, Any]:
    try:
        s3_client.delete_object(Bucket=bucket_name, Key=file_name)
        logger.info(f"File {file_name} deleted successfully.")
        return {"status": "success"}
    except Exception as e:
        return handle_minio_error(e)


def bucket_exists(bucket_name: str) -> bool:
    """Check if the MinIO bucket exists."""
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        logger.info(f"Bucket {bucket_name} exists.")
        return True
    except s3_client.exceptions.NoSuchBucket:
        logger.warning(f"Bucket {bucket_name} does not exist.")
        return False
    except Exception as e:
        logger.error(f"Error checking bucket existence: {e}")
        return False


def handle_minio_error(e: Exception) -> Dict[str, Any]:
    """Helper to handle MinIO errors."""
    if isinstance(e, s3_client.exceptions.ClientError):
        error_code = e.response['Error']['Code']
        if error_code == '404':
            return {"status": "failed", "error": "File not found"}
        else:
            logger.error(f"ClientError occurred: {e}")
            return {"status": "failed", "error": str(e)}
    else:
        logger.error(f"Unexpected error: {e}")
        return {"status": "failed", "error": str(e)}
