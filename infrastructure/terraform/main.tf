# AWS Infrastructure Setup with Terraform
# This creates the necessary AWS resources for the CI/CD pipeline

terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  
  # Uncomment for remote state (recommended for production)
  # backend "s3" {
  #   bucket = "your-terraform-state-bucket"
  #   key    = "housing-price-api/terraform.tfstate"
  #   region = "us-east-1"
  # }
}

provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = "housing-price-api"
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

# ===========================================
# Variables
# ===========================================

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "demo"
}

variable "app_name" {
  description = "Application name"
  type        = string
  default     = "housing-price-api"
}

# ===========================================
# ECR Repository
# ===========================================

resource "aws_ecr_repository" "app" {
  name                 = var.app_name
  image_tag_mutability = "MUTABLE"
  
  image_scanning_configuration {
    scan_on_push = true
  }
  
  encryption_configuration {
    encryption_type = "AES256"
  }
}

resource "aws_ecr_lifecycle_policy" "app" {
  repository = aws_ecr_repository.app.name
  
  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep last 10 images"
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = 10
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}

# ===========================================
# IAM Role for App Runner
# ===========================================

resource "aws_iam_role" "apprunner_ecr_access" {
  name = "${var.app_name}-apprunner-ecr-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "build.apprunner.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "apprunner_ecr_policy" {
  role       = aws_iam_role.apprunner_ecr_access.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess"
}

# ===========================================
# IAM Role for App Runner Instance
# ===========================================

resource "aws_iam_role" "apprunner_instance" {
  name = "${var.app_name}-apprunner-instance-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "tasks.apprunner.amazonaws.com"
        }
      }
    ]
  })
}

# Add CloudWatch Logs permissions
resource "aws_iam_role_policy" "apprunner_cloudwatch" {
  name = "${var.app_name}-cloudwatch-policy"
  role = aws_iam_role.apprunner_instance.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "*"
      }
    ]
  })
}

# ===========================================
# App Runner Service
# ===========================================

resource "aws_apprunner_service" "app" {
  service_name = "${var.app_name}-service"
  
  source_configuration {
    authentication_configuration {
      access_role_arn = aws_iam_role.apprunner_ecr_access.arn
    }
    
    image_repository {
      image_configuration {
        port = "8000"
        
        runtime_environment_variables = {
          API_HOST  = "0.0.0.0"
          API_PORT  = "8000"
          LOG_LEVEL = "INFO"
        }
      }
      
      image_identifier      = "${aws_ecr_repository.app.repository_url}:latest"
      image_repository_type = "ECR"
    }
    
    auto_deployments_enabled = true
  }
  
  instance_configuration {
    cpu               = "1024"   # 1 vCPU
    memory            = "2048"   # 2 GB
    instance_role_arn = aws_iam_role.apprunner_instance.arn
  }
  
  health_check_configuration {
    healthy_threshold   = 2
    unhealthy_threshold = 3
    interval            = 10
    path                = "/health"
    protocol            = "HTTP"
    timeout             = 5
  }
  
  auto_scaling_configuration_arn = aws_apprunner_auto_scaling_configuration_version.app.arn
  
  tags = {
    Name = "${var.app_name}-service"
  }
}

# ===========================================
# Auto Scaling Configuration
# ===========================================

resource "aws_apprunner_auto_scaling_configuration_version" "app" {
  auto_scaling_configuration_name = "${var.app_name}-autoscaling"
  
  max_concurrency = 100
  max_size        = 5
  min_size        = 1
  
  tags = {
    Name = "${var.app_name}-autoscaling"
  }
}

# ===========================================
# CloudWatch Log Group
# ===========================================

resource "aws_cloudwatch_log_group" "app" {
  name              = "/aws/apprunner/${var.app_name}"
  retention_in_days = 14
}

# ===========================================
# Outputs
# ===========================================

output "ecr_repository_url" {
  description = "ECR repository URL"
  value       = aws_ecr_repository.app.repository_url
}

output "apprunner_service_url" {
  description = "App Runner service URL"
  value       = aws_apprunner_service.app.service_url
}

output "apprunner_service_arn" {
  description = "App Runner service ARN"
  value       = aws_apprunner_service.app.arn
}

output "apprunner_role_arn" {
  description = "App Runner ECR access role ARN (for GitHub Actions)"
  value       = aws_iam_role.apprunner_ecr_access.arn
}
