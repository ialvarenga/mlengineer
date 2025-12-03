import json
import os
import pathlib
import pickle
from datetime import datetime, timezone
from typing import List
from typing import Tuple

import boto3
import pandas
from botocore.exceptions import ClientError
from sklearn import model_selection
from sklearn import neighbors
from sklearn import pipeline
from sklearn import preprocessing

SALES_PATH = "data/kc_house_data.csv"  # path to CSV with home sale data
DEMOGRAPHICS_PATH = "data/kc_house_data.csv"  # path to CSV with demographics
# List of columns (subset) that will be taken from home sale data
SALES_COLUMN_SELECTION = [
    'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'sqft_above', 'sqft_basement', 'zipcode'
]
OUTPUT_DIR = "model"  # Directory where output artifacts will be saved

# S3 Configuration
S3_MODEL_BUCKET = os.getenv("S3_MODEL_BUCKET", "phdata-housing-models")
S3_MODEL_PREFIX = os.getenv("S3_MODEL_PREFIX", "models/housing")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
UPLOAD_TO_S3 = os.getenv("UPLOAD_TO_S3", "true").lower() == "true"


def load_data(
    sales_path: str, demographics_path: str, sales_column_selection: List[str]
) -> Tuple[pandas.DataFrame, pandas.Series]:
    """Load the target and feature data by merging sales and demographics.

    Args:
        sales_path: path to CSV file with home sale data
        demographics_path: path to CSV file with home sale data
        sales_column_selection: list of columns from sales data to be used as
            features

    Returns:
        Tuple containg with two elements: a DataFrame and a Series of the same
        length.  The DataFrame contains features for machine learning, the
        series contains the target variable (home sale price).

    """
    data = pandas.read_csv(sales_path,
                           usecols=sales_column_selection,
                           dtype={'zipcode': str})
    demographics = pandas.read_csv("data/zipcode_demographics.csv",
                                   dtype={'zipcode': str})

    merged_data = data.merge(demographics, how="left",
                             on="zipcode").drop(columns="zipcode")
    # Remove the target variable from the dataframe, features will remain
    y = merged_data.pop('price')
    x = merged_data

    return x, y


def upload_to_s3(local_path: pathlib.Path, s3_key: str, bucket: str) -> bool:
    """Upload a file to S3.
    
    Args:
        local_path: Path to the local file
        s3_key: S3 object key (path in bucket)
        bucket: S3 bucket name
        
    Returns:
        True if upload succeeded, False otherwise
    """
    try:
        s3_client = boto3.client('s3', region_name=AWS_REGION)
        s3_client.upload_file(str(local_path), bucket, s3_key)
        print(f"[OK] Uploaded {local_path.name} to s3://{bucket}/{s3_key}")
        return True
    except ClientError as e:
        print(f"[ERROR] Failed to upload {local_path.name} to S3: {e}")
        return False


def update_latest_pointer(version: str, bucket: str, prefix: str) -> bool:
    """Update the 'latest' pointer file in S3 to point to the new version.
    
    Args:
        version: The version string (timestamp) to point to
        bucket: S3 bucket name
        prefix: S3 prefix for models
        
    Returns:
        True if update succeeded, False otherwise
    """
    try:
        s3_client = boto3.client('s3', region_name=AWS_REGION)
        latest_key = f"{prefix}/latest.txt"
        s3_client.put_object(
            Bucket=bucket,
            Key=latest_key,
            Body=version.encode('utf-8'),
            ContentType='text/plain'
        )
        print(f"[OK] Updated latest pointer to version: {version}")
        return True
    except ClientError as e:
        print(f"[ERROR] Failed to update latest pointer: {e}")
        return False


def main():
    """Load data, train model, and export artifacts."""
    x, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)
    x_train, _x_test, y_train, _y_test = model_selection.train_test_split(
        x, y, random_state=42)

    model = pipeline.make_pipeline(preprocessing.RobustScaler(),
                                   neighbors.KNeighborsRegressor()).fit(
                                       x_train, y_train)

    output_dir = pathlib.Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    # Generate version timestamp
    version = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    
    # Local file paths
    model_path = output_dir / "model.pkl"
    features_path = output_dir / "model_features.json"
    
    # Output model artifacts locally: pickled model and JSON list of features
    pickle.dump(model, open(model_path, 'wb'))
    json.dump(list(x_train.columns), open(features_path, 'w'))
    
    print(f"\nModel artifacts saved locally to: {output_dir}/")
    print(f"   - model.pkl")
    print(f"   - model_features.json")
    
    # Upload to S3 if enabled
    if UPLOAD_TO_S3:
        print(f"\nUploading to S3 bucket: {S3_MODEL_BUCKET}")
        print(f"   Version: {version}")
        
        # S3 keys with version
        s3_model_key = f"{S3_MODEL_PREFIX}/{version}/model.pkl"
        s3_features_key = f"{S3_MODEL_PREFIX}/{version}/model_features.json"
        
        # Upload files
        model_uploaded = upload_to_s3(model_path, s3_model_key, S3_MODEL_BUCKET)
        features_uploaded = upload_to_s3(features_path, s3_features_key, S3_MODEL_BUCKET)
        
        # Update latest pointer if both uploads succeeded
        if model_uploaded and features_uploaded:
            update_latest_pointer(version, S3_MODEL_BUCKET, S3_MODEL_PREFIX)
            print(f"\n[OK] Model version {version} successfully uploaded to S3!")
            print(f"   S3 path: s3://{S3_MODEL_BUCKET}/{S3_MODEL_PREFIX}/{version}/")
        else:
            print("\n[WARNING] Some uploads failed. Check AWS credentials and bucket permissions.")
    else:
        print("\n[SKIP] S3 upload skipped (UPLOAD_TO_S3=false)")


if __name__ == "__main__":
    main()
