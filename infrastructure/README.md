# AWS Infrastructure Setup Guide

This guide explains how to set up the AWS infrastructure for the Housing Price Prediction API.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         AWS Cloud                                    │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                     GitHub Actions                            │   │
│  │  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐   │   │
│  │  │  Test   │───▶│  Build  │───▶│  Push   │───▶│ Deploy  │   │   │
│  │  └─────────┘    └─────────┘    └─────────┘    └─────────┘   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                        │                            │
│                                        ▼                            │
│  ┌─────────────┐              ┌─────────────────┐                  │
│  │    ECR      │◀────────────▶│   App Runner    │                  │
│  │ (Container  │              │   (Auto-scale   │                  │
│  │  Registry)  │              │    1-5 inst)    │                  │
│  └─────────────┘              └────────┬────────┘                  │
│                                        │                            │
│                                        ▼                            │
│                               ┌─────────────────┐                  │
│                               │   CloudWatch    │                  │
│                               │   (Logs/Metrics)│                  │
│                               └─────────────────┘                  │
└─────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
                               ┌─────────────────┐
                               │   End Users     │
                               │  (HTTPS API)    │
                               └─────────────────┘
```

## Prerequisites

1. **AWS Account** with admin access
2. **AWS CLI** installed and configured
3. **Terraform** >= 1.0 installed
4. **GitHub Repository** with the code

## Quick Setup

### Option 1: Using Terraform (Recommended)

```bash
# Navigate to infrastructure directory
cd infrastructure/terraform

# Initialize Terraform
terraform init

# Preview changes
terraform plan

# Apply infrastructure
terraform apply
```

### Option 2: Manual AWS Console Setup

1. **Create ECR Repository**
   - Go to ECR → Create repository
   - Name: `housing-price-api`
   - Enable scan on push

2. **Create App Runner Service**
   - Go to App Runner → Create service
   - Source: ECR
   - Port: 8000
   - CPU: 1 vCPU, Memory: 2 GB
   - Health check: `/health`

## GitHub Secrets Required

Add these secrets to your GitHub repository (Settings → Secrets → Actions):

| Secret Name | Description |
|------------|-------------|
| `AWS_ACCESS_KEY_ID` | AWS IAM user access key |
| `AWS_SECRET_ACCESS_KEY` | AWS IAM user secret key |
| `APP_RUNNER_ROLE_ARN` | ARN of App Runner ECR access role |

### Creating IAM User for GitHub Actions

```bash
# Create IAM user
aws iam create-user --user-name github-actions-housing-api

# Attach required policies
aws iam attach-user-policy --user-name github-actions-housing-api \
  --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryFullAccess

aws iam attach-user-policy --user-name github-actions-housing-api \
  --policy-arn arn:aws:iam::aws:policy/AWSAppRunnerFullAccess

# Create access keys
aws iam create-access-key --user-name github-actions-housing-api
```

## Deployment Workflow

1. **Push to `main` branch**
2. **GitHub Actions triggers:**
   - Run tests
   - Build Docker image
   - Push to ECR
   - Deploy to App Runner
   - Run integration tests

3. **App Runner:**
   - Pulls new image from ECR
   - Performs rolling update
   - Health checks validate deployment

## Cost Estimation (Demo Usage)

| Service | Estimated Monthly Cost |
|---------|----------------------|
| App Runner (1 instance, minimal traffic) | ~$5-15 |
| ECR (< 1GB storage) | ~$0.10 |
| CloudWatch Logs | ~$0.50 |
| **Total** | **~$6-16/month** |

*Costs vary based on actual usage. App Runner charges per vCPU-hour and GB-hour.*

## Monitoring

### CloudWatch Metrics
- Request count
- Response latency (P50, P90, P99)
- Error rates
- CPU/Memory utilization

### Logs
```bash
# View App Runner logs
aws logs tail /aws/apprunner/housing-price-api --follow
```

## Cleanup

To avoid ongoing charges:

```bash
# Using Terraform
cd infrastructure/terraform
terraform destroy

# Or manually delete:
# 1. App Runner service
# 2. ECR repository (and images)
# 3. IAM roles
# 4. CloudWatch log groups
```

## Security Considerations

1. **HTTPS** - App Runner provides HTTPS by default
2. **IAM Roles** - Least privilege access
3. **Container Scanning** - ECR scans for vulnerabilities
4. **Secrets** - Use GitHub Secrets, never commit credentials
5. **CORS** - Restrict origins in production

## S3 Model Storage

The API can load ML models from S3, enabling:
- **Model versioning**: Each training run creates a timestamped version
- **Easy rollback**: Switch to any previous model version
- **Hot reloading**: Update models without restarting the API

### S3 Bucket Structure

```
s3://phdata-housing-models/
└── models/
    └── housing/
        ├── latest.txt              # Points to current version
        ├── 20251202153000/         # Version timestamp (YYYYMMDDHHmmss)
        │   ├── model.pkl
        │   └── model_features.json
        └── 20251201120000/
            ├── model.pkl
            └── model_features.json
```

### IAM Policy for S3 Model Access

Apply the policy from `iam-policy-s3-model.json` to allow read/write access:

```bash
# Create the policy
aws iam create-policy \
  --policy-name HousingModelS3Access \
  --policy-document file://infrastructure/iam-policy-s3-model.json

# Attach to your user or role
aws iam attach-user-policy \
  --user-name YOUR_USERNAME \
  --policy-arn arn:aws:iam::YOUR_ACCOUNT_ID:policy/HousingModelS3Access
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `USE_S3_MODEL` | Load model from S3 | `false` |
| `S3_MODEL_BUCKET` | S3 bucket name | `phdata-housing-models` |
| `S3_MODEL_PREFIX` | S3 prefix path | `models/housing` |
| `S3_MODEL_VERSION` | Specific version or `latest` | `latest` |
| `UPLOAD_TO_S3` | Upload model after training | `true` |

### Usage

```bash
# Train and upload model to S3
export UPLOAD_TO_S3=true
export S3_MODEL_BUCKET=phdata-housing-models
python create_model.py

# Run API loading model from S3
export USE_S3_MODEL=true
export S3_MODEL_VERSION=latest  # or specific: 20251202153000
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Scaling

App Runner auto-scales based on:
- Concurrent requests (max 100 per instance)
- Min instances: 1
- Max instances: 5

For production, consider:
- Increasing max instances
- Adding a CDN (CloudFront)
- Multi-region deployment
