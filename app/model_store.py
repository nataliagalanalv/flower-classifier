import hashlib
from pathlib import Path

import boto3


def sha256_of(path: Path) -> str:
    # Read in chunks so a large file is not loaded into memory at once
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_checkpoint(bucket: str, key: str, dest: Path, expected_sha256: str | None = None) -> None:
    """Download the model from a private S3 bucket. Any failure propagates (fail-fast)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3")  # credentials come from the standard boto3 chain
    s3.download_file(bucket, key, str(dest))

    if expected_sha256 is not None and sha256_of(dest).lower() != expected_sha256.lower():
        dest.unlink(missing_ok=True)  # do not leave a file that failed verification
        raise RuntimeError(f"Checksum mismatch for s3://{bucket}/{key}")