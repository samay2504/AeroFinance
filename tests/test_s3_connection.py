"""
AWS S3 Connection Test
Tests S3 connectivity and permissions with your configured credentials.
"""

import os
from dotenv import load_dotenv
load_dotenv()

import boto3
from botocore.exceptions import ClientError

# Get credentials from environment
bucket_name = os.getenv("AWS_S3_BUCKET")
region = os.getenv("AWS_REGION", "ap-south-1")
access_key = os.getenv("AWS_ACCESS_KEY_ID")
secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

print("=" * 70)
print("AWS S3 CONNECTION TEST")
print("=" * 70)
print(f"Bucket: {bucket_name}")
print(f"Region: {region}")
print(f"Access Key: {access_key[:10]}..." if access_key else "NOT SET")
print()

if not all([bucket_name, access_key, secret_key]):
    print("❌ ERROR: Missing AWS credentials in .env file")
    print("\nRequired environment variables:")
    print("  AWS_S3_BUCKET")
    print("  AWS_ACCESS_KEY_ID")
    print("  AWS_SECRET_ACCESS_KEY")
    print("  AWS_REGION (optional, defaults to ap-south-1)")
    exit(1)

try:
    # Create S3 client
    s3 = boto3.client(
        's3',
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    )
    
    # Test 1: List buckets (verify credentials)
    print("Test 1: Verifying credentials...")
    response = s3.list_buckets()
    print(f"✅ Credentials valid! Found {len(response['Buckets'])} buckets")
    bucket_names = [b['Name'] for b in response['Buckets']]
    print(f"   Buckets: {', '.join(bucket_names)}")
    
    # Test 2: Check if bucket exists
    print(f"\nTest 2: Checking if bucket '{bucket_name}' exists...")
    s3.head_bucket(Bucket=bucket_name)
    print(f"✅ Bucket exists and is accessible!")
    
    # Test 3: Get bucket location
    print(f"\nTest 3: Checking bucket location...")
    location = s3.get_bucket_location(Bucket=bucket_name)
    bucket_region = location.get('LocationConstraint') or 'us-east-1'
    print(f"✅ Bucket region: {bucket_region}")
    if bucket_region != region:
        print(f"⚠️  WARNING: Bucket region ({bucket_region}) != configured region ({region})")
        print(f"   Update AWS_REGION={bucket_region} in .env file")
    
    # Test 4: Upload a test file
    print(f"\nTest 4: Uploading test file...")
    test_content = f"Hello from Valuenaire! Testing S3 connection.\nTimestamp: {os.popen('date /t && time /t').read().strip()}"
    s3.put_object(
        Bucket=bucket_name,
        Key="test/connection-test.txt",
        Body=test_content.encode('utf-8'),
        ContentType='text/plain'
    )
    print(f"✅ Upload successful!")
    print(f"   S3 Path: s3://{bucket_name}/test/connection-test.txt")
    
    # Test 5: Download the file
    print(f"\nTest 5: Downloading test file...")
    response = s3.get_object(Bucket=bucket_name, Key="test/connection-test.txt")
    downloaded = response['Body'].read().decode('utf-8')
    print(f"✅ Download successful!")
    print(f"   Content: {downloaded[:60]}...")
    
    # Test 6: List objects in test folder
    print(f"\nTest 6: Listing objects in test/ folder...")
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix="test/", MaxKeys=10)
    if 'Contents' in response:
        print(f"✅ Found {len(response['Contents'])} objects:")
        for obj in response['Contents']:
            size_kb = obj['Size'] / 1024
            print(f"   - {obj['Key']} ({size_kb:.2f} KB)")
    else:
        print(f"✅ Folder is empty (besides our test file)")
    
    # Test 7: Delete the test file
    print(f"\nTest 7: Cleaning up test file...")
    s3.delete_object(Bucket=bucket_name, Key="test/connection-test.txt")
    print(f"✅ Cleanup successful!")
    
    print("\n" + "=" * 70)
    print("🎉 ALL TESTS PASSED - S3 IS READY!")
    print("=" * 70)
    print("\nYour application can now:")
    print("  ✅ Upload DataFrames to S3")
    print("  ✅ Download DataFrames from S3")
    print("  ✅ List and manage files")
    print("\nNext steps:")
    print("  1. Start your application: uvicorn app.main:app --reload")
    print("  2. Upload a file via API")
    print("  3. Check S3 bucket in AWS Console")
    
except ClientError as e:
    error_code = e.response['Error']['Code']
    print(f"\n❌ ERROR: {error_code}")
    print(f"Message: {e.response['Error']['Message']}")
    print("\n" + "=" * 70)
    print("TROUBLESHOOTING")
    print("=" * 70)
    
    if error_code == 'NoSuchBucket':
        print("Issue: Bucket doesn't exist or name is incorrect")
        print("\nSolutions:")
        print("  1. Check bucket name in .env file (AWS_S3_BUCKET)")
        print("  2. Verify bucket exists in AWS S3 Console")
        print(f"  3. Current bucket name: {bucket_name}")
        print("  4. Bucket names are case-sensitive and globally unique")
    
    elif error_code in ['InvalidAccessKeyId', 'SignatureDoesNotMatch']:
        print("Issue: Invalid AWS credentials")
        print("\nSolutions:")
        print("  1. Check AWS_ACCESS_KEY_ID in .env file")
        print("  2. Check AWS_SECRET_ACCESS_KEY in .env file")
        print("  3. Verify no extra spaces or newlines in credentials")
        print("  4. Regenerate keys in AWS IAM Console if needed:")
        print("     - IAM → Users → Your User → Security credentials")
        print("     - Create access key → Download CSV")
    
    elif error_code == 'AccessDenied':
        print("Issue: User doesn't have permission to access bucket")
        print("\nSolutions:")
        print("  1. Check IAM policy attached to user")
        print("  2. Required permissions: s3:PutObject, s3:GetObject, s3:ListBucket")
        print("  3. Verify bucket name in policy matches actual bucket")
        print("  4. In AWS Console:")
        print("     - IAM → Users → Your User → Permissions")
        print("     - Attach AmazonS3FullAccess (or custom policy)")
    
    elif error_code == 'InvalidBucketName':
        print("Issue: Invalid bucket name format")
        print("\nSolutions:")
        print("  1. Bucket name must be lowercase")
        print("  2. No spaces or special characters (except hyphens)")
        print("  3. 3-63 characters long")
        print("  4. Must be globally unique")
        print(f"  5. Current bucket name: {bucket_name}")
    
    else:
        print(f"Unexpected error. Check AWS documentation for error code: {error_code}")
    
    exit(1)

except Exception as e:
    print(f"\n❌ UNEXPECTED ERROR: {e}")
    print("\nPossible causes:")
    print("  1. Network connectivity issues")
    print("  2. boto3 library not installed: pip install boto3")
    print("  3. Invalid region format")
    
    import traceback
    print("\nFull traceback:")
    traceback.print_exc()
    exit(1)
