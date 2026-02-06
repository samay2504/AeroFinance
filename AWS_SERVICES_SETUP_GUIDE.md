# AWS Services Setup Guide
## For External Deployment with AWS Storage

**Use Case**: Your application runs on your own servers/hosting (not AWS ECS/Lambda), but uses AWS services for storage and caching.

---

## 📋 Overview

You will set up:
- ✅ **AWS S3** - For DataFrame/file storage
- ✅ **AWS IAM** - For access credentials
- ⚠️ **AWS ElastiCache (Optional)** - For distributed Redis cache

Your application will run **anywhere** (local, VPS, Railway, Render, etc.) and connect to these AWS services.

---

## 🎯 Prerequisites

- AWS Account (create at [aws.amazon.com](https://aws.amazon.com))
- Access to AWS Console
- Credit card on file (free tier eligible)

---

## 📦 Part 1: S3 Bucket Setup (Required)

### Step 1.1: Create S3 Bucket

1. **Login to AWS Console**
   - Go to [console.aws.amazon.com](https://console.aws.amazon.com)
   - Login with your credentials

2. **Open S3 Service**
   - Search for "S3" in the top search bar
   - Click "S3" (Scalable Storage in the Cloud)

3. **Create Bucket**
   - Click **"Create bucket"** (orange button)
   
   **Bucket Settings**:
   ```
   Bucket name: valuenaire-dataframes-prod
   (Must be globally unique, lowercase, no spaces)
   
   AWS Region: ap-south-1 (Mumbai)
   (Choose closest to your users)
   
   Object Ownership: ACLs disabled (recommended)
   
   Block Public Access: ✅ Block all public access
   (Keep this checked - your app uses credentials)
   
   Bucket Versioning: Disabled
   (Optional: Enable if you want file history)
   
   Encryption: Enable (SSE-S3)
   (Recommended for security)
   
   Object Lock: Disabled
   ```

4. **Click "Create bucket"** at bottom

5. **Verify Creation**
   - You should see your bucket in the list
   - Click on bucket name to open it
   - It will be empty initially ✅

### Step 1.2: Note Your Bucket Details

Save these values (you'll need them later):
```
Bucket Name: valuenaire-dataframes-prod
Region: ap-south-1
ARN: arn:aws:s3:::valuenaire-dataframes-prod
```

---

## 🔐 Part 2: IAM User & Credentials Setup (Required)

### Step 2.1: Create IAM User

1. **Open IAM Service**
   - Search for "IAM" in top search bar
   - Click "IAM" (Identity and Access Management)

2. **Go to Users**
   - Click "Users" in left sidebar
   - Click **"Create user"** (blue button)

3. **User Details**
   ```
   User name: valuenaire-app-user
   
   ✅ Provide user access to the AWS Management Console - optional
   (Uncheck this - we only need programmatic access)
   ```
   - Click **"Next"**

4. **Set Permissions**
   - Select: **"Attach policies directly"**
   - Search for: `AmazonS3FullAccess`
   - ✅ Check the box next to it
   - Click **"Next"**

5. **Review and Create**
   - Review settings
   - Click **"Create user"**

### Step 2.2: Create Access Keys

1. **Open Your User**
   - Click on the user name you just created
   - Go to **"Security credentials"** tab

2. **Create Access Key**
   - Scroll down to "Access keys"
   - Click **"Create access key"**

3. **Select Use Case**
   - Choose: **"Application running outside AWS"**
   - Click **"Next"**

4. **Description (Optional)**
   ```
   Description tag: Valuenaire Production App
   ```
   - Click **"Create access key"**

5. **Download Credentials** ⚠️ IMPORTANT
   ```
   Access key ID: AKIA...
   Secret access key: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
   ```
   
   - ✅ Click **"Download .csv file"** (IMPORTANT!)
   - ⚠️ This is the ONLY time you can see the secret key!
   - Store securely (password manager recommended)
   - Click **"Done"**

### Step 2.3: Create Custom IAM Policy (Recommended - Better Security)

Instead of `AmazonS3FullAccess`, create a restricted policy:

1. **Go to IAM → Policies**
   - Click **"Create policy"**

2. **JSON Editor**
   - Click "JSON" tab
   - Paste this policy:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "ValuenaireS3Access",
            "Effect": "Allow",
            "Action": [
                "s3:PutObject",
                "s3:GetObject",
                "s3:DeleteObject",
                "s3:ListBucket",
                "s3:GetBucketLocation"
            ],
            "Resource": [
                "arn:aws:s3:::valuenaire-dataframes-prod",
                "arn:aws:s3:::valuenaire-dataframes-prod/*"
            ]
        }
    ]
}
```

3. **Policy Details**
   ```
   Policy name: ValuenaireS3Policy
   Description: Restricted S3 access for Valuenaire app
   ```
   - Click **"Create policy"**

4. **Attach to User**
   - Go back to IAM → Users
   - Click your user
   - Click "Add permissions" → "Attach policies directly"
   - Search for "ValuenaireS3Policy"
   - Attach it
   - Remove "AmazonS3FullAccess" if attached

---

## 🧪 Part 3: Test S3 Connection (Verify Setup)

### Step 3.1: Update .env File

Add these to your `Re/.env` file:

```bash
# AWS S3 Storage Configuration
DEPLOYMENT_ENV=aws
AWS_S3_BUCKET=valuenaire-dataframes-prod
AWS_REGION=ap-south-1
AWS_ACCESS_KEY_ID=AKIA...              # From Step 2.2
AWS_SECRET_ACCESS_KEY=wJalr...         # From Step 2.2
```

### Step 3.2: Test Upload

Create a test script `test_s3_connection.py`:

```python
import os
from dotenv import load_dotenv
load_dotenv()

import boto3
from botocore.exceptions import ClientError

# Get credentials from environment
bucket_name = os.getenv("AWS_S3_BUCKET")
region = os.getenv("AWS_REGION")
access_key = os.getenv("AWS_ACCESS_KEY_ID")
secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

print("=" * 70)
print("AWS S3 CONNECTION TEST")
print("=" * 70)
print(f"Bucket: {bucket_name}")
print(f"Region: {region}")
print(f"Access Key: {access_key[:10]}..." if access_key else "NOT SET")
print()

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
    
    # Test 2: Check if bucket exists
    print(f"\nTest 2: Checking if bucket '{bucket_name}' exists...")
    s3.head_bucket(Bucket=bucket_name)
    print(f"✅ Bucket exists and is accessible!")
    
    # Test 3: Upload a test file
    print(f"\nTest 3: Uploading test file...")
    test_content = "Hello from Valuenaire! Testing S3 connection."
    s3.put_object(
        Bucket=bucket_name,
        Key="test/connection-test.txt",
        Body=test_content.encode('utf-8')
    )
    print(f"✅ Upload successful! File: test/connection-test.txt")
    
    # Test 4: Download the file
    print(f"\nTest 4: Downloading test file...")
    response = s3.get_object(Bucket=bucket_name, Key="test/connection-test.txt")
    downloaded = response['Body'].read().decode('utf-8')
    print(f"✅ Download successful! Content: {downloaded[:50]}...")
    
    # Test 5: Delete the test file
    print(f"\nTest 5: Cleaning up test file...")
    s3.delete_object(Bucket=bucket_name, Key="test/connection-test.txt")
    print(f"✅ Cleanup successful!")
    
    print("\n" + "=" * 70)
    print("🎉 ALL TESTS PASSED - S3 IS READY!")
    print("=" * 70)
    
except ClientError as e:
    error_code = e.response['Error']['Code']
    print(f"\n❌ ERROR: {error_code}")
    print(f"Message: {e.response['Error']['Message']}")
    print("\nTroubleshooting:")
    
    if error_code == 'NoSuchBucket':
        print("  - Bucket name doesn't exist or has typo")
        print("  - Check AWS_S3_BUCKET in .env file")
    
    elif error_code in ['InvalidAccessKeyId', 'SignatureDoesNotMatch']:
        print("  - Invalid credentials")
        print("  - Check AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
        print("  - Regenerate keys in IAM console if needed")
    
    elif error_code == 'AccessDenied':
        print("  - User doesn't have permission to this bucket")
        print("  - Check IAM policy attached to user")
        print("  - Verify bucket name is correct")
    
    elif error_code == 'InvalidBucketName':
        print("  - Invalid bucket name format")
        print("  - Must be lowercase, no spaces, 3-63 characters")
    
except Exception as e:
    print(f"\n❌ UNEXPECTED ERROR: {e}")
    import traceback
    traceback.print_exc()
```

### Step 3.3: Run Test

```powershell
cd d:\Projects2.0\Valuenaire\Re
python test_s3_connection.py
```

**Expected Output**:
```
======================================================================
AWS S3 CONNECTION TEST
======================================================================
Bucket: valuenaire-dataframes-prod
Region: ap-south-1
Access Key: AKIA...

Test 1: Verifying credentials...
✅ Credentials valid! Found 1 buckets

Test 2: Checking if bucket 'valuenaire-dataframes-prod' exists...
✅ Bucket exists and is accessible!

Test 3: Uploading test file...
✅ Upload successful! File: test/connection-test.txt

Test 4: Downloading test file...
✅ Download successful! Content: Hello from Valuenaire! Testing S3 connect...

Test 5: Cleaning up test file...
✅ Cleanup successful!

======================================================================
🎉 ALL TESTS PASSED - S3 IS READY!
======================================================================
```

---

## 🔴 Part 4: ElastiCache Redis Setup (Optional - Advanced)

**Note**: ElastiCache requires VPC setup and is more complex. **Skip this if**:
- You're using Upstash Redis (already configured in your .env) ✅
- You don't need multi-instance caching
- Your app runs on a single server

### When You DO Need ElastiCache:
- Multiple application servers/containers
- Need low-latency cache across instances
- High traffic production deployment

### Step 4.1: Create ElastiCache Cluster

1. **Open ElastiCache Console**
   - Search for "ElastiCache" in AWS Console
   - Click "ElastiCache"

2. **Create Redis Cluster**
   - Click "Create" → "Redis cluster"
   
   **Cluster Settings**:
   ```
   Cluster mode: Disabled (smaller/simpler)
   Cluster info:
     Name: valuenaire-redis-cache
     Description: Redis cache for Valuenaire app
   
   Location: AWS Cloud
   Multi-AZ: No (for cost savings, Yes for production)
   
   Cluster settings:
     Engine version: 7.0
     Port: 6379
     Parameter group: default.redis7
     Node type: cache.t3.micro (free tier eligible)
     Number of replicas: 0 (1+ for production)
   
   Subnet group: Create new
     Name: valuenaire-subnet-group
     VPC: (select your default VPC)
     Subnets: Select all available
   
   Security:
     Security groups: Create new or select default
     Encryption at rest: Enabled (recommended)
     Encryption in transit: Enabled (recommended)
   ```

3. **Configure Security Group**
   - After creation, note the security group ID
   - Go to EC2 → Security Groups
   - Find the ElastiCache security group
   - Edit inbound rules:
   ```
   Type: Custom TCP
   Port: 6379
   Source: Your server IP or 0.0.0.0/0 (less secure)
   Description: Redis access for Valuenaire
   ```

4. **Get Endpoint**
   - Once cluster is "Available" (takes 5-10 minutes)
   - Click on cluster name
   - Copy "Primary endpoint":
   ```
   valuenaire-redis-cache.abc123.cache.amazonaws.com:6379
   ```

### Step 4.2: Update .env for ElastiCache

```bash
# Replace your Upstash URL with ElastiCache endpoint
REDIS_URL=redis://valuenaire-redis-cache.abc123.cache.amazonaws.com:6379

# Or if using TLS:
REDIS_URL=rediss://valuenaire-redis-cache.abc123.cache.amazonaws.com:6379
```

### Step 4.3: Test Redis Connection

Create `test_redis_connection.py`:

```python
import os
from dotenv import load_dotenv
load_dotenv()

try:
    import redis
    
    redis_url = os.getenv("REDIS_URL")
    print(f"Testing connection to: {redis_url}")
    
    client = redis.from_url(redis_url, socket_timeout=5)
    
    # Test SET
    client.set("test:connection", "success")
    
    # Test GET
    value = client.get("test:connection")
    
    # Test DELETE
    client.delete("test:connection")
    
    print("✅ Redis connection successful!")
    
except Exception as e:
    print(f"❌ Redis connection failed: {e}")
```

---

## 📊 Part 5: Verify Application Integration

### Step 5.1: Start Your Application

```powershell
cd d:\Projects2.0\Valuenaire\Re
uvicorn app.main:app --reload
```

### Step 5.2: Upload a Test File

```powershell
curl -F "file=@test.xlsx" http://localhost:8000/api/v1/ingest/upload
```

### Step 5.3: Check S3 Bucket

1. Go to AWS S3 Console
2. Open your bucket: `valuenaire-dataframes-prod`
3. You should see folder structure:
   ```
   dataframes/
     ├── client_xxx/
     │   └── doc_yyy/
     │       └── Sheet1.parquet
   ```

### Step 5.4: Query the Data

```powershell
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"dataset_id":"doc_yyy","query":"What is the revenue?"}'
```

**Your app will**:
- ✅ Load DataFrame from S3
- ✅ Cache schema in Redis
- ✅ Use semantic caching for responses
- ✅ Return result

---

## 💰 Cost Estimation (AWS Free Tier)

### S3 Storage
```
Free Tier: 5 GB storage, 20,000 GET requests, 2,000 PUT requests/month
Typical usage (100 files, 10MB each): ~$0.02/month
After free tier: $0.023/GB/month
```

### ElastiCache Redis
```
Free Tier: 750 hours/month of cache.t2.micro (1 year only)
After free tier: cache.t3.micro ~$13/month
Alternative: Use Upstash (free tier: 10K commands/day) ✅ Already configured
```

### Data Transfer
```
Free: 100 GB outbound/month (first 12 months)
After: $0.09/GB
```

**Recommendation**: Start with S3 + Upstash Redis (mostly free) ✅

---

## 🔒 Security Best Practices

### 1. Restrict IAM Permissions
```json
{
  "Effect": "Allow",
  "Action": ["s3:PutObject", "s3:GetObject", "s3:DeleteObject"],
  "Resource": "arn:aws:s3:::your-bucket/*"
}
```
✅ Don't use `s3:*` or `AmazonS3FullAccess` in production

### 2. Use Environment Variables (Never Hardcode)
```bash
# ❌ BAD - in code
bucket = "valuenaire-prod"

# ✅ GOOD - from environment
bucket = os.getenv("AWS_S3_BUCKET")
```

### 3. Enable S3 Bucket Versioning
- Protects against accidental deletion
- Can recover previous versions

### 4. Enable Server-Side Encryption
- S3 → Bucket → Properties → Default encryption
- Use SSE-S3 (free) or SSE-KMS (more control)

### 5. Rotate Access Keys Regularly
- IAM → Users → Security credentials
- Create new key → Update .env → Delete old key
- Do this every 90 days

### 6. Use AWS Secrets Manager (Production)
Instead of .env file:
```python
import boto3
client = boto3.client('secretsmanager')
secret = client.get_secret_value(SecretId='valuenaire/prod/aws-keys')
```

---

## 🐛 Troubleshooting Common Issues

### Error: "NoSuchBucket"
```
✓ Check bucket name spelling in .env
✓ Verify bucket exists in S3 console
✓ Check region matches in .env
```

### Error: "AccessDenied"
```
✓ Verify IAM user has correct permissions
✓ Check policy is attached to user
✓ Confirm bucket name in policy matches actual bucket
```

### Error: "InvalidAccessKeyId"
```
✓ Verify access key ID is correct (no extra spaces)
✓ Check if key was deleted in IAM console
✓ Regenerate keys if needed
```

### Error: "SignatureDoesNotMatch"
```
✓ Verify secret access key is correct
✓ Check for hidden characters when copying
✓ Regenerate keys and try again
```

### Error: "Connection timeout" (ElastiCache)
```
✓ Check security group allows port 6379
✓ Verify your server IP is whitelisted
✓ Confirm VPC/subnet configuration
✓ Use VPN if connecting from outside AWS
```

---

## 📋 Summary Checklist

### AWS Console Setup (One-Time):
- [ ] Create S3 bucket
- [ ] Create IAM user
- [ ] Attach S3 permissions policy
- [ ] Generate access keys
- [ ] Download credentials CSV
- [ ] (Optional) Create ElastiCache Redis cluster
- [ ] (Optional) Configure security groups

### Local Configuration:
- [ ] Add AWS credentials to .env file
- [ ] Run `test_s3_connection.py` (all tests pass)
- [ ] (Optional) Run `test_redis_connection.py`
- [ ] Start application
- [ ] Upload test file
- [ ] Verify file appears in S3 bucket
- [ ] Query data successfully

### Production Deployment:
- [ ] Use restricted IAM policies
- [ ] Enable S3 bucket versioning
- [ ] Enable S3 encryption
- [ ] Set up CloudWatch alarms
- [ ] Configure backup/disaster recovery
- [ ] Rotate access keys regularly

---

## 🚀 Part 6: AWS Container Deployment (Optional - Advanced)

**Note**: This section is for deploying your **entire application** on AWS infrastructure. Skip if you're running on your own servers and only using AWS S3/Redis.

### Overview: ECS vs EKS vs Fargate

| Feature | ECS (EC2) | ECS (Fargate) | EKS | Best For |
|---------|-----------|---------------|-----|----------|
| **Infrastructure** | You manage EC2 instances | AWS manages servers | You manage Kubernetes | Complex orchestration |
| **Kubernetes** | No | No | Yes | Teams using K8s |
| **Complexity** | Medium | Low | High | Depends on team |
| **Cost** | Lower (if optimized) | Higher (serverless premium) | Higher (control plane $0.10/hr) | Budget matters |
| **Scaling** | Manual/Auto-scaling | Automatic | Automatic | Traffic patterns |
| **Cold Start** | No | Yes (~5s) | No | Latency requirements |
| **Setup Time** | 1-2 hours | 30 minutes | 2-4 hours | Time to market |

**Recommendation for Valuenaire**:
- **Start with**: ECS + Fargate (easiest, no server management)
- **Scale to**: ECS + EC2 (cost optimization at scale)
- **Consider EKS if**: Already using Kubernetes, multi-cloud strategy

---

## 🐳 Part 6A: ECS with Fargate (Recommended for Beginners)

**What is it?**: Run containers without managing servers. AWS handles everything.

### Step 6A.1: Create Dockerfile

Create `Dockerfile` in your `Re/` directory:

```dockerfile
# Base image with Python 3.11
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Step 6A.2: Create ECR Repository (Docker Registry)

1. **Open ECR Console**
   - Search for "ECR" (Elastic Container Registry)
   - Click "Create repository"

2. **Repository Settings**
   ```
   Visibility: Private
   Repository name: valuenaire-app
   Tag immutability: Disabled
   Scan on push: Enabled (recommended)
   Encryption: AES-256
   ```
   - Click **"Create repository"**

3. **Note the Repository URI**
   ```
   123456789012.dkr.ecr.ap-south-1.amazonaws.com/valuenaire-app
   ```

### Step 6A.3: Build and Push Docker Image

```powershell
# 1. Authenticate Docker to ECR
aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.ap-south-1.amazonaws.com

# 2. Build Docker image
cd d:\Projects2.0\Valuenaire\Re
docker build -t valuenaire-app .

# 3. Tag image
docker tag valuenaire-app:latest 123456789012.dkr.ecr.ap-south-1.amazonaws.com/valuenaire-app:latest

# 4. Push to ECR
docker push 123456789012.dkr.ecr.ap-south-1.amazonaws.com/valuenaire-app:latest
```

### Step 6A.4: Create ECS Cluster

1. **Open ECS Console**
   - Search for "ECS" (Elastic Container Service)
   - Click "Clusters" → **"Create cluster"**

2. **Cluster Configuration**
   ```
   Cluster name: valuenaire-prod-cluster
   Infrastructure: AWS Fargate (serverless)
   Monitoring: Container Insights (optional, extra cost)
   ```
   - Click **"Create"**

### Step 6A.5: Create Task Definition

1. **Go to Task Definitions**
   - Click "Task definitions" → **"Create new task definition"**

2. **Task Definition Family**
   ```
   Task definition family: valuenaire-task
   ```

3. **Infrastructure**
   ```
   Launch type: AWS Fargate
   Operating system: Linux
   CPU: 0.5 vCPU (or 1 vCPU for production)
   Memory: 1 GB (or 2 GB for production)
   Task role: Create new role (or select existing)
   Task execution role: ecsTaskExecutionRole
   ```

4. **Container Definition**
   ```
   Container name: valuenaire-container
   Image URI: 123456789012.dkr.ecr.ap-south-1.amazonaws.com/valuenaire-app:latest
   
   Port mappings:
     Container port: 8000
     Protocol: TCP
     App protocol: HTTP
   
   Environment variables:
     DEPLOYMENT_ENV = aws
     AWS_S3_BUCKET = valuenaire-dataframes-prod
     AWS_REGION = ap-south-1
     HOT_CACHE_SEMANTIC = true
     HOT_CACHE_SIMILARITY = 0.92
     REDIS_URL = redis://your-elasticache-endpoint:6379
   
   Secrets (from AWS Secrets Manager - recommended):
     GOOGLE_API_KEY = arn:aws:secretsmanager:...
     GROQ_API_KEY = arn:aws:secretsmanager:...
     JINA_API_KEY = arn:aws:secretsmanager:...
   
   Health check:
     Command: CMD-SHELL, curl -f http://localhost:8000/health || exit 1
     Interval: 30 seconds
     Timeout: 5 seconds
     Retries: 3
     Start period: 60 seconds
   
   Logging:
     Log driver: awslogs
     Log group: /ecs/valuenaire-task
     Region: ap-south-1
     Stream prefix: ecs
   ```

5. **IAM Task Role Policy**
   
   The task role needs S3 access. Create/attach this policy:
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Allow",
         "Action": [
           "s3:GetObject",
           "s3:PutObject",
           "s3:DeleteObject",
           "s3:ListBucket"
         ],
         "Resource": [
           "arn:aws:s3:::valuenaire-dataframes-prod",
           "arn:aws:s3:::valuenaire-dataframes-prod/*"
         ]
       }
     ]
   }
   ```

6. **Click "Create"**

### Step 6A.6: Create ECS Service

1. **In Your Cluster**
   - Click on cluster name → "Services" tab
   - Click **"Create"**

2. **Deployment Configuration**
   ```
   Compute options: Launch type
   Launch type: FARGATE
   Platform version: LATEST
   
   Application type: Service
   Task definition:
     Family: valuenaire-task
     Revision: LATEST
   
   Service name: valuenaire-service
   Desired tasks: 2 (for high availability)
   ```

3. **Networking**
   ```
   VPC: (select your default VPC)
   Subnets: Select 2+ subnets in different AZs
   Security group: Create new
     Name: valuenaire-ecs-sg
     Inbound rules:
       - Type: HTTP (80)
       - Type: Custom TCP (8000)
       - Source: 0.0.0.0/0 (or restrict to ALB)
   Public IP: Enabled (for internet access)
   ```

4. **Load Balancing (Recommended)**
   ```
   Load balancer type: Application Load Balancer
   Create new load balancer:
     Name: valuenaire-alb
     Scheme: Internet-facing
   
   Target group:
     Name: valuenaire-tg
     Protocol: HTTP
     Port: 8000
     Health check path: /health
     Health check interval: 30 seconds
   ```

5. **Auto Scaling (Optional)**
   ```
   Service auto scaling: Enabled
   Minimum tasks: 1
   Maximum tasks: 10
   
   Scaling policy:
     Type: Target tracking
     Metric: ECSServiceAverageCPUUtilization
     Target value: 70
   ```

6. **Click "Create"**

### Step 6A.7: Get Application URL

1. **Wait for Service to Start** (2-5 minutes)
   - Go to ECS → Clusters → Your Cluster → Services
   - Wait for "Running count" to match "Desired count"

2. **Get Load Balancer URL**
   - Click on service → "Networking" tab
   - Copy "DNS name" of load balancer:
   ```
   http://valuenaire-alb-123456789.ap-south-1.elb.amazonaws.com
   ```

3. **Test Your API**
   ```powershell
   curl http://valuenaire-alb-123456789.ap-south-1.elb.amazonaws.com/health
   
   # Upload file
   curl -F "file=@test.xlsx" http://valuenaire-alb-123456789.ap-south-1.elb.amazonaws.com/api/v1/ingest/upload
   ```

---

## ⚡ Part 6B: ECS with EC2 (Cost Optimization)

**When to use**: Higher traffic, need cost optimization, already familiar with EC2.

### Key Differences from Fargate:
- You manage EC2 instances (patching, scaling)
- Lower cost per container (~30-50% cheaper)
- More control over instance types
- No cold starts

### Step 6B.1: Create ECS Cluster with EC2

1. **Create Cluster**
   ```
   Cluster name: valuenaire-ec2-cluster
   Infrastructure: Amazon EC2 instances
   
   EC2 instance type: t3.medium (2 vCPU, 4 GB RAM)
   Desired capacity: 2 instances
   SSH key pair: (select your key)
   ```

2. **Auto Scaling Group**
   ```
   Minimum: 1
   Maximum: 10
   VPC: Default
   Subnets: Select 2+ AZs
   ```

3. **Rest is similar to Fargate**, but:
   - Task definition: Select "EC2" launch type
   - Service: Select "EC2" launch type

### Cost Comparison (Running 24/7):
```
Fargate (0.5 vCPU, 1 GB):     ~$36/month per task
EC2 t3.medium (2 vCPU, 4 GB): ~$30/month (runs 4-6 tasks)

Break-even: ~5+ tasks → EC2 is cheaper
```

---

## ☸️ Part 6C: Amazon EKS (Kubernetes)

**When to use**: 
- Team already uses Kubernetes
- Need multi-cloud portability
- Complex microservices architecture
- GitOps workflows (ArgoCD, Flux)

### Step 6C.1: Install Required Tools

```powershell
# Install eksctl (EKS CLI)
choco install eksctl

# Install kubectl
choco install kubernetes-cli

# Verify installations
eksctl version
kubectl version --client
```

### Step 6C.2: Create EKS Cluster

```powershell
# Create cluster (takes 15-20 minutes)
eksctl create cluster \
  --name valuenaire-eks-cluster \
  --region ap-south-1 \
  --nodegroup-name valuenaire-nodes \
  --node-type t3.medium \
  --nodes 2 \
  --nodes-min 1 \
  --nodes-max 4 \
  --managed
```

### Step 6C.3: Create Kubernetes Deployment

Create `k8s/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: valuenaire-deployment
  namespace: default
spec:
  replicas: 2
  selector:
    matchLabels:
      app: valuenaire
  template:
    metadata:
      labels:
        app: valuenaire
    spec:
      containers:
      - name: valuenaire
        image: 123456789012.dkr.ecr.ap-south-1.amazonaws.com/valuenaire-app:latest
        ports:
        - containerPort: 8000
        env:
        - name: DEPLOYMENT_ENV
          value: "aws"
        - name: AWS_S3_BUCKET
          value: "valuenaire-dataframes-prod"
        - name: AWS_REGION
          value: "ap-south-1"
        - name: HOT_CACHE_SEMANTIC
          value: "true"
        - name: GOOGLE_API_KEY
          valueFrom:
            secretKeyRef:
              name: valuenaire-secrets
              key: google-api-key
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: valuenaire-service
spec:
  type: LoadBalancer
  selector:
    app: valuenaire
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
```

### Step 6C.4: Deploy to EKS

```powershell
# Create Kubernetes secrets
kubectl create secret generic valuenaire-secrets \
  --from-literal=google-api-key=AIza... \
  --from-literal=groq-api-key=gsk_... \
  --from-literal=jina-api-key=jina_...

# Apply deployment
kubectl apply -f k8s/deployment.yaml

# Check status
kubectl get pods
kubectl get svc

# Get load balancer URL
kubectl get svc valuenaire-service
```

### Step 6C.5: Set Up Ingress (Optional - Better than LoadBalancer)

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: valuenaire-ingress
  annotations:
    kubernetes.io/ingress.class: alb
    alb.ingress.kubernetes.io/scheme: internet-facing
spec:
  rules:
  - http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: valuenaire-service
            port:
              number: 80
```

---

## 💰 Cost Comparison (All Options)

### Monthly Costs (24/7 operation, ap-south-1 region)

| Option | vCPU | Memory | Instances | Monthly Cost | Notes |
|--------|------|--------|-----------|--------------|-------|
| **ECS Fargate** | 0.5 | 1 GB | 2 tasks | ~$72 | Easiest, no servers |
| **ECS EC2** (t3.medium) | 2 | 4 GB | 2 instances | ~$60 | More control |
| **EKS** | Control plane + nodes | N/A | 2 nodes | ~$145 | $72 (control) + $60 (nodes) + $13 (NAT) |
| **External (VPS)** | N/A | N/A | 1 server | $10-50 | Cheapest if low traffic |

**Additional Costs (All Options)**:
- S3: ~$1-5/month
- ElastiCache (t3.micro): ~$13/month (or use Upstash free tier)
- Data transfer: ~$1-10/month
- Load balancer (ALB): ~$23/month (only if using ALB)

**Recommendation**:
- **Dev/Testing**: External VPS + S3 (~$15/month)
- **Production (low traffic)**: Fargate (~$100/month all included)
- **Production (high traffic)**: ECS EC2 (~$100/month)
- **Enterprise**: EKS (~$200/month + scale costs)

---

## 🔄 CI/CD Pipeline (GitHub Actions Example)

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to ECS

on:
  push:
    branches: [main]

env:
  AWS_REGION: ap-south-1
  ECR_REPOSITORY: valuenaire-app
  ECS_CLUSTER: valuenaire-prod-cluster
  ECS_SERVICE: valuenaire-service
  CONTAINER_NAME: valuenaire-container

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - name: Checkout code
      uses: actions/checkout@v3

    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: ${{ env.AWS_REGION }}

    - name: Login to Amazon ECR
      id: login-ecr
      uses: aws-actions/amazon-ecr-login@v1

    - name: Build, tag, and push image to Amazon ECR
      id: build-image
      env:
        ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
        IMAGE_TAG: ${{ github.sha }}
      run: |
        cd Re
        docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
        docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
        echo "image=$ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG" >> $GITHUB_OUTPUT

    - name: Download task definition
      run: |
        aws ecs describe-task-definition \
          --task-definition valuenaire-task \
          --query taskDefinition > task-definition.json

    - name: Fill in the new image ID in the Amazon ECS task definition
      id: task-def
      uses: aws-actions/amazon-ecs-render-task-definition@v1
      with:
        task-definition: task-definition.json
        container-name: ${{ env.CONTAINER_NAME }}
        image: ${{ steps.build-image.outputs.image }}

    - name: Deploy Amazon ECS task definition
      uses: aws-actions/amazon-ecs-deploy-task-definition@v1
      with:
        task-definition: ${{ steps.task-def.outputs.task-definition }}
        service: ${{ env.ECS_SERVICE }}
        cluster: ${{ env.ECS_CLUSTER }}
        wait-for-service-stability: true
```

---

## 📊 Monitoring & Observability

### CloudWatch Logs

All container logs automatically go to CloudWatch:
```
Log group: /ecs/valuenaire-task
```

**View logs**:
1. CloudWatch Console → Log groups
2. Search for queries, errors
3. Set up alerts for ERROR level logs

### Metrics to Monitor

```
ECS/Fargate Metrics:
- CPUUtilization (target: <70%)
- MemoryUtilization (target: <80%)
- TaskCount (running vs desired)

Application Metrics:
- Request count
- Response time (P50, P95, P99)
- Error rate
- Cache hit rate (from HotPromptCache.stats())
```

### Set Up Alarms

```powershell
# CPU High alarm
aws cloudwatch put-metric-alarm \
  --alarm-name valuenaire-high-cpu \
  --alarm-description "Alert when CPU > 80%" \
  --metric-name CPUUtilization \
  --namespace AWS/ECS \
  --statistic Average \
  --period 300 \
  --threshold 80 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 2
```

---

## 🔒 Security Checklist for Container Deployment

- [ ] Use IAM roles (not access keys) for ECS tasks
- [ ] Store secrets in AWS Secrets Manager
- [ ] Enable encryption at rest (ECR, S3, EBS)
- [ ] Use VPC with private subnets
- [ ] Restrict security groups (least privilege)
- [ ] Enable Container Insights
- [ ] Scan Docker images for vulnerabilities
- [ ] Use read-only root filesystem (where possible)
- [ ] Run containers as non-root user
- [ ] Enable CloudTrail for audit logs

---

## 🎯 Next Steps

1. **Complete Part 1-2** (S3 + IAM) - Required
2. **Run Part 3** (Test connection) - Verify setup
3. **Choose deployment strategy**:
   - Run on your servers → Skip Part 6
   - Deploy to AWS → Follow Part 6A (Fargate recommended)
4. **Start your app** - It will auto-use S3
5. **Monitor costs** - Check AWS Billing dashboard

**Your application will now**:
- ✅ Store all DataFrames in AWS S3
- ✅ Use Redis for caching (Upstash or ElastiCache)
- ✅ Run anywhere (your server, VPS, cloud hosting, or AWS containers)
- ✅ Scale horizontally (multiple instances share S3/Redis)

