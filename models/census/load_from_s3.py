import boto3
import pandas as pd
import io

s3 = boto3.client("s3")
bucket = "atrium-census-data-bucket"
key = "data/philadelphia_parcels_enriched.parquet"

obj = s3.get_object(Bucket=bucket, Key=key)
df = pd.read_parquet(io.BytesIO(obj['Body'].read()))

print(df.head())