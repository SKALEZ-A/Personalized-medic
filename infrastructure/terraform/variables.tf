# AI Personalized Medicine Platform - Terraform Variables

# AWS Configuration
variable "aws_region" {
  description = "AWS region for infrastructure deployment"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name (staging, production)"
  type        = string
  validation {
    condition     = contains(["staging", "production"], var.environment)
    error_message = "Environment must be either 'staging' or 'production'."
  }
}

# VPC Configuration
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  description = "List of availability zones"
  type        = list(string)
  default     = ["us-east-1a", "us-east-1b", "us-east-1c"]
}

variable "public_subnets" {
  description = "CIDR blocks for public subnets"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}

variable "private_subnets" {
  description = "CIDR blocks for private subnets"
  type        = list(string)
  default     = ["10.0.10.0/24", "10.0.11.0/24", "10.0.12.0/24"]
}

variable "database_subnets" {
  description = "CIDR blocks for database subnets"
  type        = list(string)
  default     = ["10.0.20.0/24", "10.0.21.0/24", "10.0.22.0/24"]
}

# ECS Configuration
variable "api_desired_count" {
  description = "Desired number of API service tasks"
  type        = number
  default     = 2
}

variable "api_container_image" {
  description = "Docker image for API service"
  type        = string
  default     = "ghcr.io/your-org/healthcare-platform:latest-api"
}

variable "api_container_port" {
  description = "Port for API container"
  type        = number
  default     = 8000
}

variable "frontend_desired_count" {
  description = "Desired number of frontend service tasks"
  type        = number
  default     = 2
}

variable "frontend_container_image" {
  description = "Docker image for frontend service"
  type        = string
  default     = "ghcr.io/your-org/healthcare-platform:latest-frontend"
}

variable "frontend_container_port" {
  description = "Port for frontend container"
  type        = number
  default     = 80
}

# RDS Configuration
variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.r6g.large"
}

variable "db_engine" {
  description = "Database engine"
  type        = string
  default     = "postgres"
}

variable "db_engine_version" {
  description = "Database engine version"
  type        = string
  default     = "15.4"
}

variable "db_name" {
  description = "Database name"
  type        = string
  default     = "healthcare_platform"
}

variable "db_username" {
  description = "Database username"
  type        = string
  default     = "healthcare_admin"
  sensitive   = true
}

variable "db_password" {
  description = "Database password"
  type        = string
  sensitive   = true
}

variable "db_allocated_storage" {
  description = "Allocated storage for database (GB)"
  type        = number
  default     = 100
}

variable "db_multi_az" {
  description = "Enable Multi-AZ deployment"
  type        = bool
  default     = true
}

variable "db_backup_retention" {
  description = "Backup retention period (days)"
  type        = number
  default     = 30
}

# ElastiCache Configuration
variable "redis_node_type" {
  description = "Redis node type"
  type        = string
  default     = "cache.r6g.large"
}

variable "redis_num_cache_nodes" {
  description = "Number of Redis cache nodes"
  type        = number
  default     = 2
}

# Domain and SSL
variable "domain_name" {
  description = "Domain name for the application"
  type        = string
  default     = "healthcare-platform.com"
}

variable "certificate_arn" {
  description = "ARN of ACM certificate"
  type        = string
}

# Monitoring Configuration
variable "enable_detailed_monitoring" {
  description = "Enable detailed CloudWatch monitoring"
  type        = bool
  default     = true
}

variable "log_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 30
}

# Backup Configuration
variable "backup_retention_days" {
  description = "Backup retention period in days"
  type        = number
  default     = 30
}

variable "enable_cross_region_backup" {
  description = "Enable cross-region backup"
  type        = bool
  default     = true
}

# Cost Optimization
variable "enable_spot_instances" {
  description = "Enable spot instances for cost optimization"
  type        = bool
  default     = false
}

# High Availability
variable "enable_multi_az" {
  description = "Enable Multi-AZ deployment for high availability"
  type        = bool
  default     = true
}

# Security
variable "enable_waf" {
  description = "Enable Web Application Firewall"
  type        = bool
  default     = true
}

variable "enable_shield" {
  description = "Enable AWS Shield for DDoS protection"
  type        = bool
  default     = true
}

# Performance
variable "api_cpu_allocation" {
  description = "CPU allocation for API service (CPU units)"
  type        = number
  default     = 1024
}

variable "api_memory_allocation" {
  description = "Memory allocation for API service (MB)"
  type        = number
  default     = 2048
}

variable "frontend_cpu_allocation" {
  description = "CPU allocation for frontend service (CPU units)"
  type        = number
  default     = 512
}

variable "frontend_memory_allocation" {
  description = "Memory allocation for frontend service (MB)"
  type        = number
  default     = 1024
}

# Scaling Configuration
variable "api_min_capacity" {
  description = "Minimum capacity for API service auto-scaling"
  type        = number
  default     = 2
}

variable "api_max_capacity" {
  description = "Maximum capacity for API service auto-scaling"
  type        = number
  default     = 20
}

variable "frontend_min_capacity" {
  description = "Minimum capacity for frontend service auto-scaling"
  type        = number
  default     = 2
}

variable "frontend_max_capacity" {
  description = "Maximum capacity for frontend service auto-scaling"
  type        = number
  default     = 10
}

variable "cpu_utilization_threshold" {
  description = "CPU utilization threshold for auto-scaling"
  type        = number
  default     = 70
}

variable "memory_utilization_threshold" {
  description = "Memory utilization threshold for auto-scaling"
  type        = number
  default     = 80
}

# Disaster Recovery
variable "enable_disaster_recovery" {
  description = "Enable disaster recovery configuration"
  type        = bool
  default     = true
}

variable "dr_region" {
  description = "Disaster recovery region"
  type        = string
  default     = "us-west-2"
}

variable "rto_minutes" {
  description = "Recovery Time Objective in minutes"
  type        = number
  default     = 60
}

variable "rpo_minutes" {
  description = "Recovery Point Objective in minutes"
  type        = number
  default     = 15
}

# Compliance and Security
variable "enable_encryption" {
  description = "Enable encryption for data at rest"
  type        = bool
  default     = true
}

variable "kms_key_rotation" {
  description = "Enable KMS key rotation"
  type        = bool
  default     = true
}

variable "enable_cloudtrail" {
  description = "Enable CloudTrail for audit logging"
  type        = bool
  default     = true
}

variable "vpc_flow_logs" {
  description = "Enable VPC flow logs"
  type        = bool
  default     = true
}

# Tagging
variable "project_tags" {
  description = "Additional tags for resources"
  type        = map(string)
  default     = {}
}

# Cost Allocation
variable "cost_center" {
  description = "Cost center for cost allocation"
  type        = string
  default     = "healthcare-platform"
}

variable "business_unit" {
  description = "Business unit for resource organization"
  type        = string
  default     = "medical-research"
}
