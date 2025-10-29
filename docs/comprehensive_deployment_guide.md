# Comprehensive Deployment Guide
## AI Personalized Medicine Platform

This guide provides detailed instructions for deploying the AI Personalized Medicine Platform across different environments and scenarios.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development Setup](#local-development-setup)
3. [Staging Deployment](#staging-deployment)
4. [Production Deployment](#production-deployment)
5. [Infrastructure as Code](#infrastructure-as-code)
6. [CI/CD Pipeline](#cicd-pipeline)
7. [Monitoring & Observability](#monitoring--observability)
8. [Backup & Disaster Recovery](#backup--disaster-recovery)
9. [Scaling & Performance](#scaling--performance)
10. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

#### Development Environment
- **OS**: macOS 12+, Ubuntu 20.04+, Windows 11 (WSL2)
- **CPU**: 4+ cores recommended
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 50GB free space
- **Network**: Stable internet connection

#### Production Environment
- **Infrastructure**: AWS ECS with Fargate
- **Database**: Amazon RDS PostgreSQL
- **Cache**: Amazon ElastiCache Redis
- **Storage**: Amazon S3
- **CDN**: Amazon CloudFront
- **Monitoring**: AWS CloudWatch, X-Ray

### Software Dependencies

#### Required Tools
```bash
# Python 3.11+
python --version

# Node.js 18+
node --version
npm --version

# Docker & Docker Compose
docker --version
docker-compose --version

# AWS CLI
aws --version

# Terraform
terraform --version

# kubectl (for Kubernetes deployments)
kubectl version --client
```

#### Python Dependencies
```bash
# Install core dependencies
pip install fastapi uvicorn pydantic sqlalchemy psycopg2-binary redis aiofiles python-multipart

# Install AI/ML dependencies
pip install tensorflow torch scikit-learn pandas numpy matplotlib seaborn

# Install testing dependencies
pip install pytest pytest-cov pytest-xdist pytest-mock pytest-asyncio locust

# Install development dependencies
pip install black flake8 mypy bandit safety pre-commit
```

#### Node.js Dependencies
```bash
# Install frontend dependencies
npm install react react-dom react-router-dom framer-motion axios recharts react-responsive

# Install PWA dependencies
npm install workbox-webpack-plugin webpack-pwa-manifest

# Install testing dependencies
npm install --save-dev jest @testing-library/react @testing-library/jest-dom cypress
```

## Local Development Setup

### 1. Clone and Initialize

```bash
# Clone the repository
git clone https://github.com/your-org/healthcare-platform.git
cd healthcare-platform

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
npm install

# Copy environment configuration
cp .env.example .env
# Edit .env with your local configuration
```

### 2. Database Setup

```bash
# Start PostgreSQL with Docker
docker run --name healthcare-postgres \
  -e POSTGRES_DB=healthcare_platform \
  -e POSTGRES_USER=healthcare_admin \
  -e POSTGRES_PASSWORD=secure_password \
  -p 5432:5432 \
  -d postgres:15

# Start Redis
docker run --name healthcare-redis \
  -p 6379:6379 \
  -d redis:7-alpine

# Run database migrations
python scripts/init_database.py
```

### 3. Start Development Services

```bash
# Start the API server
python main.py

# In another terminal, start the frontend
cd frontend
npm start

# Optional: Start monitoring stack
docker-compose -f monitoring/docker-compose.yml up -d
```

### 4. Verify Installation

```bash
# Test API endpoints
curl http://localhost:8000/api/health

# Test frontend
curl http://localhost:3000

# Run basic tests
pytest tests/unit_tests/ -v
npm test
```

## Staging Deployment

### Automated Deployment (Recommended)

```bash
# Deploy to staging using GitHub Actions
git tag staging-v1.0.0
git push origin staging-v1.0.0

# Or trigger manually via GitHub UI
# Go to Actions → CI/CD Pipeline → Run workflow
# Select environment: staging
```

### Manual Deployment

#### 1. Build Docker Images

```bash
# Build API image
docker build -f deployment/Dockerfile.api -t healthcare-platform-api:staging .

# Build frontend image
docker build -f deployment/Dockerfile.frontend -t healthcare-platform-frontend:staging .

# Tag and push to registry
docker tag healthcare-platform-api:staging your-registry.com/healthcare-platform-api:staging
docker tag healthcare-platform-frontend:staging your-registry.com/healthcare-platform-frontend:staging
docker push your-registry.com/healthcare-platform-api:staging
docker push your-registry.com/healthcare-platform-frontend:staging
```

#### 2. Deploy to ECS

```bash
# Update ECS service
aws ecs update-service \
  --cluster healthcare-platform-staging \
  --service api-service \
  --force-new-deployment \
  --task-definition healthcare-platform-api-staging

# Wait for deployment
aws ecs wait services-stable \
  --cluster healthcare-platform-staging \
  --services api-service
```

#### 3. Run Smoke Tests

```bash
# Test staging endpoints
curl -f https://staging-api.healthcare-platform.com/api/health
curl -f https://staging.healthcare-platform.com
```

## Production Deployment

### Automated Production Deployment

```bash
# Create production release
git checkout main
git pull origin main
git tag production-v1.0.0
git push origin production-v1.0.0

# GitHub Actions will handle the rest
# - Build and push images
# - Run comprehensive tests
# - Deploy to production with canary strategy
# - Monitor and validate deployment
```

### Blue-Green Deployment Strategy

```bash
# 1. Deploy to blue environment
aws ecs update-service \
  --cluster healthcare-platform-production \
  --service api-service-blue \
  --force-new-deployment

# 2. Run comprehensive tests
python scripts/production_tests.py --environment blue

# 3. Switch traffic to blue
aws elbv2 modify-listener \
  --listener-arn $LISTENER_ARN \
  --default-actions Type=forward,TargetGroupArn=$BLUE_TARGET_GROUP

# 4. Keep green environment as rollback option
# 5. Scale down old environment after validation period
```

### Canarying Strategy

```bash
# Deploy 10% of traffic to new version
aws ecs update-service \
  --cluster healthcare-platform-production \
  --service api-service \
  --desired-count 22  # 10% of 200 total capacity

# Monitor metrics for 30 minutes
aws cloudwatch get-metric-statistics \
  --namespace AWS/ECS \
  --metric-name CPUUtilization \
  --start-time $(date -u +%Y-%m-%dT%H:%M:%S -d '30 minutes ago') \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 300 \
  --statistics Average

# Gradually increase traffic if metrics are good
aws ecs update-service \
  --cluster healthcare-platform-production \
  --service api-service \
  --desired-count 200  # Full capacity
```

## Infrastructure as Code

### Terraform Deployment

```bash
# Initialize Terraform
cd infrastructure/terraform
terraform init

# Plan deployment
terraform plan -var-file=staging.tfvars

# Apply changes
terraform apply -var-file=staging.tfvars

# For production
terraform apply -var-file=production.tfvars
```

### Environment-Specific Configurations

#### Staging Configuration (`staging.tfvars`)
```hcl
environment = "staging"
api_desired_count = 2
frontend_desired_count = 2
db_instance_class = "db.t3.medium"
redis_node_type = "cache.t3.micro"
enable_detailed_monitoring = true
```

#### Production Configuration (`production.tfvars`)
```hcl
environment = "production"
api_desired_count = 10
frontend_desired_count = 5
db_instance_class = "db.r6g.large"
redis_node_type = "cache.r6g.large"
enable_detailed_monitoring = true
enable_multi_az = true
enable_cross_region_backup = true
```

## CI/CD Pipeline

### Pipeline Stages

1. **Code Quality**
   - Linting (flake8, ESLint)
   - Type checking (mypy)
   - Security scanning (Bandit, Safety)
   - Code coverage (>80%)

2. **Unit Tests**
   - API endpoint tests
   - Business logic tests
   - Component tests
   - Parallel execution with pytest-xdist

3. **Integration Tests**
   - API workflow tests
   - Database integration tests
   - External service integration tests
   - WebSocket tests

4. **Performance Tests** (Optional)
   - Load testing with Locust
   - Stress testing
   - Endurance testing

5. **Security Tests**
   - Dependency vulnerability scanning
   - Container image scanning
   - Infrastructure security checks

6. **Build & Package**
   - Docker image building
   - Multi-stage builds for optimization
   - Image signing and scanning

7. **Deploy**
   - Blue-green deployments
   - Canarying strategy
   - Automated rollback on failure

8. **Monitor & Validate**
   - Health checks
   - Performance validation
   - User acceptance testing

### Customizing the Pipeline

#### Adding New Test Stages
```yaml
# In .github/workflows/ci-cd-pipeline.yml
- name: Custom Security Tests
  run: |
    # Add your custom security tests
    python scripts/custom_security_tests.py

- name: Performance Regression Tests
  run: |
    # Compare performance against baseline
    python scripts/performance_regression.py
```

#### Environment-Specific Deployments
```yaml
# Add new environment
deploy-custom-environment:
  if: github.event.inputs.environment == 'custom'
  steps:
    - name: Deploy to Custom Environment
      run: |
        # Custom deployment logic
        ./scripts/deploy-custom.sh
```

## Monitoring & Observability

### Setting Up Monitoring

```bash
# Deploy monitoring stack
cd monitoring
docker-compose up -d prometheus grafana

# Configure CloudWatch
aws logs create-log-group --log-group-name healthcare-platform-api
aws logs create-log-group --log-group-name healthcare-platform-frontend

# Set up X-Ray tracing
aws xray create-group --group-name healthcare-platform
```

### Key Metrics to Monitor

#### Application Metrics
```python
# Response time percentiles
response_time_p50 < 200ms
response_time_p95 < 500ms
response_time_p99 < 1000ms

# Error rates
api_error_rate < 1%
authentication_failure_rate < 0.1%

# Throughput
requests_per_second > 1000
active_connections < 10000
```

#### Infrastructure Metrics
```python
# ECS metrics
cpu_utilization < 80%
memory_utilization < 85%

# RDS metrics
database_connections < 80% of max
read_iops < 80% of provisioned
write_iops < 80% of provisioned

# ElastiCache metrics
cache_hit_rate > 95%
evictions = 0
```

#### Business Metrics
```python
# User engagement
daily_active_users > 1000
session_duration > 300 seconds

# Feature usage
genomic_analyses_per_day > 100
appointments_scheduled_per_day > 50

# Data quality
invalid_records_percentage < 0.01%
data_processing_time < 30 seconds
```

### Alert Configuration

```yaml
# Critical alerts
api_down:
  condition: up == 0
  labels:
    severity: critical
  annotations:
    summary: "API service is down"

high_error_rate:
  condition: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
  labels:
    severity: warning
  annotations:
    summary: "High error rate detected"

# Performance alerts
slow_response_time:
  condition: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1.0
  labels:
    severity: warning
  annotations:
    summary: "Slow response times detected"
```

## Backup & Disaster Recovery

### Automated Backups

#### Database Backups
```bash
# Configure automated RDS backups
aws rds modify-db-instance \
  --db-instance-identifier healthcare-platform-prod \
  --backup-retention-period 30 \
  --preferred-backup-window 03:00-04:00 \
  --enable-cloudwatch-logs-exports '["postgresql"]'

# Cross-region backup replication
aws rds create-db-instance-read-replica \
  --db-instance-identifier healthcare-platform-dr \
  --source-db-instance-identifier healthcare-platform-prod \
  --source-region us-east-1 \
  --region us-west-2
```

#### File System Backups
```bash
# S3 bucket versioning and replication
aws s3api put-bucket-versioning \
  --bucket healthcare-platform-data \
  --versioning-configuration Status=Enabled

# Cross-region replication
aws s3api put-bucket-replication \
  --bucket healthcare-platform-data \
  --replication-configuration file://replication-config.json
```

### Disaster Recovery Testing

#### Regular DR Drills
```bash
# Monthly DR test script
#!/bin/bash
echo "Starting Disaster Recovery Drill"

# 1. Simulate failure
echo "Simulating primary region failure..."
aws ec2 stop-instances --instance-ids $PRIMARY_INSTANCES

# 2. Initiate failover
echo "Initiating failover to DR region..."
./scripts/failover-to-dr.sh

# 3. Validate recovery
echo "Validating DR recovery..."
./scripts/validate-dr-recovery.sh

# 4. Failback preparation
echo "Preparing for failback..."
./scripts/prepare-failback.sh

echo "DR Drill completed"
```

#### Recovery Time/Objective Validation
```bash
# Measure RTO/RPO compliance
python scripts/validate-rto-rpo.py \
  --incident-start "2024-01-15T10:00:00Z" \
  --recovery-complete "2024-01-15T10:45:00Z" \
  --last-backup "2024-01-15T09:45:00Z"
```

## Scaling & Performance

### Auto-Scaling Configuration

#### ECS Service Auto-Scaling
```bash
# CPU-based scaling
aws application-autoscaling put-scaling-policy \
  --policy-name cpu-scaling \
  --service-namespace ecs \
  --resource-id service/healthcare-platform-production/api-service \
  --scalable-dimension ecs:service:DesiredCount \
  --policy-type TargetTrackingScaling \
  --target-tracking-scaling-policy-configuration '{
    "TargetValue": 70.0,
    "PredefinedMetricSpecification": {
      "PredefinedMetricType": "ECSServiceAverageCPUUtilization"
    }
  }'

# Request-based scaling
aws application-autoscaling put-scaling-policy \
  --policy-name request-scaling \
  --service-namespace ecs \
  --resource-id service/healthcare-platform-production/api-service \
  --scalable-dimension ecs:service:DesiredCount \
  --policy-type TargetTrackingScaling \
  --target-tracking-scaling-policy-configuration '{
    "TargetValue": 1000.0,
    "PredefinedMetricSpecification": {
      "PredefinedMetricType": "ALBRequestCountPerTarget"
    }
  }'
```

#### RDS Read Replica Scaling
```bash
# Add read replicas for read-heavy workloads
aws rds create-db-instance-read-replica \
  --db-instance-identifier healthcare-platform-replica-1 \
  --source-db-instance-identifier healthcare-platform-prod

# Configure read replica auto-scaling
aws application-autoscaling register-scalable-target \
  --service-namespace rds \
  --resource-id db-cluster:healthcare-platform-cluster \
  --scalable-dimension rds:cluster:ReadReplicaCount \
  --min-capacity 1 \
  --max-capacity 5
```

### Performance Optimization

#### Database Optimization
```sql
-- Create indexes for common queries
CREATE INDEX idx_patient_medical_history ON patients USING GIN (medical_history);
CREATE INDEX idx_health_metrics_patient_timestamp ON health_metrics (patient_id, timestamp DESC);
CREATE INDEX idx_appointments_patient_date ON appointments (patient_id, scheduled_time);

-- Partition large tables
CREATE TABLE health_metrics_y2024m01 PARTITION OF health_metrics
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
```

#### Caching Strategy
```python
# Multi-level caching configuration
CACHING_CONFIG = {
    'redis_ttl': {
        'patient_data': 3600,      # 1 hour
        'health_metrics': 300,     # 5 minutes
        'appointments': 1800,      # 30 minutes
        'reports': 7200           # 2 hours
    },
    'cdn_ttl': {
        'static_assets': 86400,   # 24 hours
        'api_responses': 300,     # 5 minutes
        'dynamic_content': 60     # 1 minute
    }
}
```

#### CDN Configuration
```bash
# CloudFront behavior configuration
aws cloudfront create-distribution \
  --distribution-config '{
    "CallerReference": "healthcare-platform-cdn",
    "Comment": "Healthcare Platform CDN",
    "Enabled": true,
    "Origins": {
      "Quantity": 1,
      "Items": [{
        "Id": "healthcare-platform-frontend",
        "DomainName": "healthcare-platform-frontend.s3.amazonaws.com",
        "S3OriginConfig": {
          "OriginAccessIdentity": ""
        }
      }]
    },
    "CacheBehaviors": {
      "Quantity": 2,
      "Items": [
        {
          "PathPattern": "/api/*",
          "TargetOriginId": "healthcare-platform-api",
          "ViewerProtocolPolicy": "redirect-to-https",
          "CachePolicyId": "api-cache-policy"
        },
        {
          "PathPattern": "/static/*",
          "TargetOriginId": "healthcare-platform-frontend",
          "ViewerProtocolPolicy": "redirect-to-https",
          "CachePolicyId": "static-cache-policy"
        }
      ]
    }
  }'
```

## Troubleshooting

### Common Issues and Solutions

#### API Service Issues

**High Response Times**
```bash
# Check ECS service status
aws ecs describe-services --cluster healthcare-platform-production --services api-service

# Check CloudWatch metrics
aws cloudwatch get-metric-statistics \
  --namespace AWS/ECS \
  --metric-name CPUUtilization \
  --start-time $(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 300 \
  --statistics Average

# Check application logs
aws logs tail /aws/ecs/healthcare-platform-api --follow
```

**Database Connection Issues**
```bash
# Check RDS status
aws rds describe-db-instances --db-instance-identifier healthcare-platform-prod

# Check database connections
psql -h $DB_HOST -U $DB_USER -d healthcare_platform -c "SELECT count(*) FROM pg_stat_activity;"

# Reset database connections if needed
aws rds reboot-db-instance --db-instance-identifier healthcare-platform-prod
```

#### Frontend Issues

**CDN Issues**
```bash
# Check CloudFront distribution status
aws cloudfront get-distribution --id $DISTRIBUTION_ID

# Invalidate cache if needed
aws cloudfront create-invalidation --distribution-id $DISTRIBUTION_ID --paths "/*"

# Check S3 bucket permissions
aws s3api get-bucket-policy --bucket healthcare-platform-frontend
```

**JavaScript Errors**
```javascript
// Enable detailed error logging
window.addEventListener('error', function(e) {
  console.error('JavaScript Error:', e.error);
  // Send to error tracking service
});

// Check network requests
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.addEventListener('message', function(event) {
    if (event.data.type === 'ERROR') {
      console.error('Service Worker Error:', event.data.error);
    }
  });
}
```

#### Deployment Issues

**ECS Deployment Failures**
```bash
# Check deployment status
aws ecs describe-services --cluster healthcare-platform-production --services api-service

# Check task logs
TASK_ARN=$(aws ecs list-tasks --cluster healthcare-platform-production --service-name api-service --query 'taskArns[0]' --output text)
aws ecs describe-tasks --cluster healthcare-platform-production --tasks $TASK_ARN

# Check container logs
aws logs tail /aws/ecs/healthcare-platform-api --follow
```

**Docker Image Issues**
```bash
# Check image size and layers
docker history healthcare-platform-api:latest

# Validate image security
docker scan healthcare-platform-api:latest

# Test image locally
docker run -p 8000:8000 healthcare-platform-api:latest
```

### Debugging Tools

#### Application Debugging
```bash
# Enable debug mode
export DEBUG=true
export LOG_LEVEL=DEBUG

# Start with debugging enabled
python main.py --debug --reload

# Use debugger
python -m pdb main.py
```

#### Database Debugging
```sql
-- Check slow queries
SELECT pid, now() - pg_stat_activity.query_start AS duration, query
FROM pg_stat_activity
WHERE state = 'active'
ORDER BY duration DESC;

-- Check table sizes
SELECT schemaname, tablename, pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Check index usage
SELECT indexrelname, idx_scan, idx_tup_read, idx_tup_fetch
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;
```

#### Network Debugging
```bash
# Check connectivity
curl -v https://api.healthcare-platform.com/api/health

# Check DNS resolution
nslookup api.healthcare-platform.com

# Check SSL certificate
openssl s_client -connect api.healthcare-platform.com:443 -servername api.healthcare-platform.com

# Network packet capture (if accessible)
tcpdump -i eth0 host api.healthcare-platform.com -w capture.pcap
```

### Performance Profiling

#### Application Profiling
```python
import cProfile
import pstats

def profile_function(func):
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()

        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        stats.print_stats(20)  # Top 20 functions

        return result
    return wrapper

# Usage
@profile_function
def process_health_data(data):
    # Function implementation
    pass
```

#### System Profiling
```bash
# CPU profiling
perf record -a -g -- sleep 60
perf report

# Memory profiling
valgrind --tool=massif --stacks=yes python main.py
ms_print massif.out.*

# Network profiling
iftop -i eth0
```

---

## Appendices

### A. Environment Variables
### B. Service Dependencies
### C. Monitoring Dashboards
### D. Backup Verification Scripts
### E. Performance Benchmarks
### F. Security Checklist
### G. Compliance Requirements
