import os
import boto3
from botocore.exceptions import ClientError
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# Fetch necessary AWS/S3-compatible credentials and information
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
AWS_DEFAULT_REGION = os.environ.get('AWS_DEFAULT_REGION')
AWS_S3_BUCKET = os.environ.get('AWS_S3_BUCKET')
AWS_S3_ENDPOINT = os.environ.get('AWS_S3_ENDPOINT')
AWS_PATH = os.environ.get('AWS_PATH', '').strip('/')

# Destination base directory
DESTINATION = "/opt/app-root/src/hf_home/"

VERIFY_SSL = os.getenv("S3_VERIFY_SSL")

verify_ssl_certs = True

if VERIFY_SSL:
    # Do not verify SSL certificates
    USE_CERTS = not VERIFY_SSL.lower() in ["0", "false"]
    verify_ssl_certs = False
else:
    # Verify SSL certificates, by default
    USE_CERTS = True

if USE_CERTS:
    AWS_CA_BUNDLE = os.getenv("AWS_CA_BUNDLE")
    # Should verify certs, but no cert specified
    if not AWS_CA_BUNDLE:
        logging.error(f"SSL verification enabled, but no certificate path found")
        sys.exit(1)
    else:
        # Should verify certs, cert specified, but not found
        if not os.path.exists(AWS_CA_BUNDLE):
            logging.error(f"SSL CA bundle not found at {AWS_CA_BUNDLE}")
            sys.exit(1)
        else:
            verify_ssl_certs = AWS_CA_BUNDLE


class S3Assets:
    def __init__(self,
                 bucket: str,
                 prefix: str,
                 destination: str, ):
        self._client = boto3.client(
            's3',
            endpoint_url=AWS_S3_ENDPOINT,
            region_name=AWS_DEFAULT_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            verify=verify_ssl_certs
        )
        self.bucket = bucket
        self.prefix = prefix
        self.destination = destination

    @staticmethod
    def get_local_values(destination, basename):
        lmeval_file = os.path.join(destination, basename)
        lmeval_folder = os.path.dirname(lmeval_file)
        return lmeval_file, lmeval_folder

    @staticmethod
    def _get_basename(key, prefix) -> str:
        if key.startswith(prefix):
            return key[len(prefix):]
        else:
            return key

    def process_object(self, _object):
        key = _object['Key']

        prefix = self.prefix if self.prefix.endswith('/') else self.prefix + '/'

        basename_path = self._get_basename(key, prefix)

        if key.endswith('/'):
            return

        lmeval_file, lmeval_folder = self.get_local_values(self.destination, basename_path)

        if not os.path.exists(lmeval_folder):
            os.makedirs(lmeval_folder, exist_ok=True)

        logging.info(f"Downloading s3://{self.bucket}/{key} -> {lmeval_file}")
        self._client.download_file(self.bucket, key, lmeval_file)

    def process_page(self, page):
        if 'Contents' not in page:
            logging.error(f"No objects found with prefix '{self.prefix}' in bucket '{self.bucket}'.")
            return

        for _object in page['Contents']:
            self.process_object(_object)

    def download(self) -> None:
        """
        Download the contents of a bucket, with a specific prefix, locally.
        """
        paginator = self._client.get_paginator('list_objects_v2')
        try:
            for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
                self.process_page(page)
        except ClientError as e:
            logging.error(f"Error: {e}")


if __name__ == "__main__":
    s3Assets = S3Assets(bucket=AWS_S3_BUCKET, prefix=AWS_PATH, destination=DESTINATION)
    s3Assets.download()
