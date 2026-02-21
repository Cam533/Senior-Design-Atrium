#!/usr/bin/env python3
"""
Test script to verify S3 and RDS connectivity for the image upload feature.
Run this before going live with image uploads.

Usage:
  python test_image_setup.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment
_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(_env_path)

def test_rds_connection():
    """Test RDS database connection and plot_images table."""
    print("\n" + "="*60)
    print("Testing RDS Connection")
    print("="*60)
    
    try:
        sys.path.append(str(Path(__file__).resolve().parent))
        from access.db_access import get_db_engine
        
        engine = get_db_engine("atrium_census")
        if engine is None:
            print("❌ RDS engine is None. Check AWS credentials in .env")
            return False
        
        # Try to query plot_images table
        from sqlalchemy import text
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1 FROM plot_images LIMIT 1"))
            result.fetchone()
        
        print("✅ RDS Connection: OK")
        print("✅ plot_images table exists")
        return True
    
    except Exception as e:
        error_msg = str(e).lower()
        if "plot_images" in error_msg or "does not exist" in error_msg:
            print("❌ plot_images table not found")
            print("   Run: python access/create_plot_images_table.py")
        else:
            print(f"❌ RDS Connection Failed: {e}")
        return False


def test_s3_connection():
    """Test S3 bucket access and permissions."""
    print("\n" + "="*60)
    print("Testing S3 Connection")
    print("="*60)
    
    try:
        import boto3
        from botocore.exceptions import ClientError
        
        # Check environment variables
        access_key = os.getenv("AWS_ACCESS_KEY_ID")
        secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        bucket = os.getenv("S3_BUCKET_NAME")
        region = os.getenv("AWS_REGION", "us-east-1")
        
        if not all([access_key, secret_key, bucket]):
            print("❌ Missing AWS credentials in .env:")
            print(f"   AWS_ACCESS_KEY_ID: {'✓' if access_key else '✗'}")
            print(f"   AWS_SECRET_ACCESS_KEY: {'✓' if secret_key else '✗'}")
            print(f"   S3_BUCKET_NAME: {'✓' if bucket else '✗'}")
            return False
        
        # Try to connect
        s3_client = boto3.client(
            "s3",
            region_name=region,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )
        
        # Try to list bucket
        response = s3_client.list_objects_v2(Bucket=bucket, MaxKeys=1)
        
        print("✅ S3 Connection: OK")
        print(f"   Bucket: {bucket}")
        print(f"   Region: {region}")
        print(f"   Objects in bucket: {response.get('KeyCount', 0)}")
        return True
    
    except ImportError:
        print("❌ boto3 not installed")
        print("   Run: pip install boto3")
        return False
    
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        print(f"❌ S3 Error: {error_code}")
        print(f"   Message: {e}")
        
        if error_code == "InvalidAccessKeyId":
            print("   Check AWS_ACCESS_KEY_ID in .env")
        elif error_code == "SignatureDoesNotMatch":
            print("   Check AWS_SECRET_ACCESS_KEY in .env")
        elif error_code == "NoSuchBucket":
            print("   Check S3_BUCKET_NAME in .env")
        
        return False
    
    except Exception as e:
        print(f"❌ S3 Connection Failed: {e}")
        return False


def test_database_schema():
    """Check the plot_images table schema."""
    print("\n" + "="*60)
    print("Testing Database Schema")
    print("="*60)
    
    try:
        sys.path.append(str(Path(__file__).resolve().parent))
        from access.db_access import get_db_engine
        from sqlalchemy import text
        
        engine = get_db_engine("atrium_census")
        if engine is None:
            return False
        
        with engine.connect() as conn:
            # Check table exists and get schema
            result = conn.execute(text("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = 'plot_images'
                ORDER BY ordinal_position
            """))
            
            columns = result.fetchall()
            if not columns:
                print("❌ plot_images table not found")
                return False
            
            print("✅ plot_images table schema:")
            for col_name, col_type in columns:
                print(f"   - {col_name}: {col_type}")
            
            return True
    
    except Exception as e:
        print(f"❌ Schema check failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("Image Upload Feature Setup Verification")
    print("="*60)
    
    results = {
        "RDS Connection": test_rds_connection(),
        "Database Schema": test_database_schema(),
        "S3 Connection": test_s3_connection(),
    }
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    all_passed = all(results.values())
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print("\n" + "="*60)
    
    if all_passed:
        print("✅ All checks passed! Ready for image uploads.")
        print("\nNext steps:")
        print("1. Start the backend:  python -m uvicorn backend.main:app --reload")
        print("2. Start the frontend: cd frontend && npm run dev")
        print("3. Click a parcel on the map to test image upload")
    else:
        print("❌ Some checks failed. See above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
