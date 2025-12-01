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

## Scaling

App Runner auto-scales based on:
- Concurrent requests (max 100 per instance)
- Min instances: 1
- Max instances: 5

For production, consider:
- Increasing max instances
- Adding a CDN (CloudFront)
- Multi-region deployment
