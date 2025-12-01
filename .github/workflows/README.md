# CI/CD Pipeline Setup

This document explains the CI/CD pipeline configuration for the Housing Price Prediction API.

## Pipeline Overview

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│    TEST      │────▶│    BUILD     │────▶│    DEPLOY    │────▶│  INTEGRATION │
│              │     │              │     │              │     │    TESTS     │
│ • Lint       │     │ • Docker     │     │ • App Runner │     │ • API Tests  │
│ • Unit Tests │     │ • Push ECR   │     │ • Auto-scale │     │ • Health     │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

## Pipeline Stages

### 1. Test & Lint (`test`)
**Triggers:** All pushes and PRs

- Checks out code
- Sets up Python 3.11
- Installs dependencies
- Runs flake8 linting
- Checks code formatting (black)
- Trains model for tests
- Runs pytest unit tests
- Uploads model artifacts

### 2. Build & Push (`build`)
**Triggers:** Only `main` branch, after tests pass

- Downloads model artifacts
- Configures AWS credentials
- Logs into Amazon ECR
- Builds Docker image
- Tags with commit SHA and `latest`
- Pushes to ECR

### 3. Deploy (`deploy`)
**Triggers:** Only `main` branch, after build

- Configures AWS credentials
- Deploys to AWS App Runner
- Waits for service stability
- Outputs service URL
- Performs initial health check

### 4. Integration Tests (`integration-test`)
**Triggers:** After successful deployment

- Tests live API endpoints
- Validates health endpoint
- Tests prediction endpoint

## GitHub Secrets Configuration

Navigate to: **Repository → Settings → Secrets and variables → Actions**

Add the following secrets:

```
AWS_ACCESS_KEY_ID       = <your-aws-access-key>
AWS_SECRET_ACCESS_KEY   = <your-aws-secret-key>
APP_RUNNER_ROLE_ARN     = arn:aws:iam::<account-id>:role/housing-price-api-apprunner-ecr-role
```

## Local Testing

Test the pipeline locally before pushing:

```bash
# Run linting
flake8 app/ ml/ --count --select=E9,F63,F7,F82 --show-source --statistics

# Run tests
pytest tests/ -v

# Build Docker image locally
docker build -t housing-price-api -f docker/Dockerfile .

# Run container locally
docker run -p 8000:8000 housing-price-api
```

## Deployment Environments

| Branch | Environment | Auto-Deploy |
|--------|-------------|-------------|
| `main` | Production  | Yes         |
| `develop` | Staging   | No (tests only) |
| Feature branches | None | No (tests only) |

## Rollback Procedure

If a deployment fails:

1. **Automatic:** App Runner keeps previous version running if health checks fail
2. **Manual rollback:**
   ```bash
   # List recent images
   aws ecr describe-images --repository-name housing-price-api --query 'imageDetails[*].imageTags'
   
   # Deploy specific version
   aws apprunner update-service --service-arn <arn> --source-configuration imageRepository={imageIdentifier=<ecr-url>:<tag>}
   ```

## Monitoring Deployments

```bash
# Check App Runner service status
aws apprunner describe-service --service-arn <arn>

# View deployment logs
aws logs tail /aws/apprunner/housing-price-api --follow

# Check service URL
aws apprunner describe-service --service-arn <arn> --query 'Service.ServiceUrl'
```

## Cost Optimization

- App Runner scales to zero when not in use (if configured)
- ECR lifecycle policy keeps only last 10 images
- CloudWatch logs retention set to 14 days
