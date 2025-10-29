# AI Personalized Medicine Platform Infrastructure
# Terraform configuration for AWS infrastructure

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  backend "s3" {
    bucket         = "healthcare-platform-terraform-state"
    key            = "infrastructure/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "healthcare-platform-terraform-locks"
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "AI-Personalized-Medicine-Platform"
      Environment = var.environment
      ManagedBy   = "Terraform"
      Owner       = "Healthcare-Platform-Team"
    }
  }
}

# VPC Configuration
module "vpc" {
  source = "./modules/vpc"

  environment         = var.environment
  vpc_cidr           = var.vpc_cidr
  availability_zones = var.availability_zones
  public_subnets     = var.public_subnets
  private_subnets    = var.private_subnets
  database_subnets   = var.database_subnets
}

# Security Groups
module "security_groups" {
  source = "./modules/security-groups"

  environment = var.environment
  vpc_id      = module.vpc.vpc_id

  depends_on = [module.vpc]
}

# Load Balancer
module "alb" {
  source = "./modules/alb"

  environment         = var.environment
  vpc_id              = module.vpc.vpc_id
  public_subnet_ids   = module.vpc.public_subnet_ids
  alb_security_groups = [module.security_groups.alb_sg_id]
  certificate_arn     = var.certificate_arn

  depends_on = [module.vpc, module.security_groups]
}

# ECS Cluster and Services
module "ecs" {
  source = "./modules/ecs"

  environment           = var.environment
  vpc_id                = module.vpc.vpc_id
  private_subnet_ids    = module.vpc.private_subnet_ids
  ecs_security_groups   = [module.security_groups.ecs_sg_id]
  alb_target_group_arn  = module.alb.api_target_group_arn
  alb_listener_arn      = module.alb.alb_listener_arn

  # API Service Configuration
  api_desired_count    = var.api_desired_count
  api_container_image  = var.api_container_image
  api_container_port   = var.api_container_port

  # Frontend Service Configuration
  frontend_desired_count   = var.frontend_desired_count
  frontend_container_image = var.frontend_container_image
  frontend_container_port  = var.frontend_container_port

  depends_on = [module.vpc, module.security_groups, module.alb]
}

# RDS Database
module "rds" {
  source = "./modules/rds"

  environment            = var.environment
  vpc_id                 = module.vpc.vpc_id
  database_subnet_ids    = module.vpc.database_subnet_ids
  rds_security_groups    = [module.security_groups.rds_sg_id]
  db_instance_class      = var.db_instance_class
  db_engine             = var.db_engine
  db_engine_version     = var.db_engine_version
  db_name               = var.db_name
  db_username           = var.db_username
  db_password           = var.db_password
  db_allocated_storage  = var.db_allocated_storage
  db_multi_az           = var.db_multi_az
  db_backup_retention   = var.db_backup_retention

  depends_on = [module.vpc, module.security_groups]
}

# ElastiCache (Redis)
module "elasticache" {
  source = "./modules/elasticache"

  environment               = var.environment
  vpc_id                    = module.vpc.vpc_id
  private_subnet_ids        = module.vpc.private_subnet_ids
  elasticache_security_groups = [module.security_groups.elasticache_sg_id]
  redis_node_type           = var.redis_node_type
  redis_num_cache_nodes     = var.redis_num_cache_nodes

  depends_on = [module.vpc, module.security_groups]
}

# S3 Buckets
module "s3" {
  source = "./modules/s3"

  environment = var.environment

  depends_on = []
}

# CloudWatch Monitoring
module "cloudwatch" {
  source = "./modules/cloudwatch"

  environment = var.environment

  depends_on = []
}

# WAF (Web Application Firewall)
module "waf" {
  source = "./modules/waf"

  environment = var.environment
  alb_arn     = module.alb.alb_arn

  depends_on = [module.alb]
}

# Route 53 DNS
module "route53" {
  source = "./modules/route53"

  environment    = var.environment
  domain_name    = var.domain_name
  alb_dns_name   = module.alb.alb_dns_name
  alb_zone_id    = module.alb.alb_zone_id
  certificate_arn = var.certificate_arn

  depends_on = [module.alb]
}

# CloudFront CDN (for frontend)
module "cloudfront" {
  source = "./modules/cloudfront"

  environment               = var.environment
  domain_name              = var.domain_name
  s3_bucket_regional_domain = module.s3.frontend_bucket_domain
  certificate_arn          = var.certificate_arn

  depends_on = [module.s3]
}

# Backup and Disaster Recovery
module "backup" {
  source = "./modules/backup"

  environment = var.environment
  rds_arn     = module.rds.rds_arn
  s3_bucket_arn = module.s3.backup_bucket_arn

  depends_on = [module.rds, module.s3]
}

# Outputs
output "vpc_id" {
  description = "VPC ID"
  value       = module.vpc.vpc_id
}

output "alb_dns_name" {
  description = "ALB DNS name"
  value       = module.alb.alb_dns_name
}

output "api_service_name" {
  description = "ECS API service name"
  value       = module.ecs.api_service_name
}

output "frontend_service_name" {
  description = "ECS Frontend service name"
  value       = module.ecs.frontend_service_name
}

output "rds_endpoint" {
  description = "RDS endpoint"
  value       = module.rds.rds_endpoint
  sensitive   = true
}

output "redis_endpoint" {
  description = "Redis endpoint"
  value       = module.elasticache.redis_endpoint
  sensitive   = true
}

output "cloudfront_domain" {
  description = "CloudFront distribution domain"
  value       = module.cloudfront.cloudfront_domain
}
