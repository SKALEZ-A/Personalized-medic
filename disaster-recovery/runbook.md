# Disaster Recovery Runbook
## AI Personalized Medicine Platform

### Overview
This runbook provides procedures for disaster recovery scenarios affecting the AI Personalized Medicine Platform. The platform implements a multi-region, multi-AZ architecture designed for high availability and rapid recovery.

### Recovery Objectives

| Component | RTO | RPO | Recovery Method |
|-----------|-----|-----|-----------------|
| API Services | 15 minutes | 5 minutes | Automated failover |
| Database | 30 minutes | 15 minutes | Cross-region replication |
| Frontend | 5 minutes | 0 minutes | CDN failover |
| Data Lake | 60 minutes | 30 minutes | Cross-region backup |

### Emergency Contacts

- **Primary On-Call**: healthcare-platform-oncall@company.com
- **Secondary On-Call**: infrastructure-team@company.com
- **Management**: platform-managers@company.com
- **Legal/Compliance**: compliance@company.com
- **External Support**: AWS Premium Support (1-888-280-4331)

---

## Incident Response Procedures

### Phase 1: Detection & Assessment (0-5 minutes)

#### 1.1 Alert Detection
- **Automated Monitoring**: AWS CloudWatch alarms trigger alerts
- **Manual Detection**: Team members report issues via Slack/Email
- **External Monitoring**: Third-party monitoring services

#### 1.2 Initial Assessment
```bash
# Check system status
aws ecs describe-services --cluster healthcare-platform-production --services api-service
aws rds describe-db-instances --db-instance-identifier healthcare-platform-prod
aws elbv2 describe-load-balancers --load-balancer-arns $ALB_ARN

# Check application health
curl -f https://api.healthcare-platform.com/health
curl -f https://healthcare-platform.com/health
```

#### 1.3 Severity Classification
- **SEV-0**: Complete system outage (>50% users affected)
- **SEV-1**: Major functionality broken (25-50% users affected)
- **SEV-2**: Minor functionality degraded (5-25% users affected)
- **SEV-3**: Non-critical issue (<5% users affected)

---

### Phase 2: Containment (5-15 minutes)

#### 2.1 Isolate Affected Systems
```bash
# Scale down affected services
aws ecs update-service \
  --cluster healthcare-platform-production \
  --service api-service \
  --desired-count 0

# Enable maintenance mode
aws s3 cp maintenance.html s3://healthcare-platform-frontend/maintenance.html
```

#### 2.2 Traffic Diversion
```bash
# Route traffic to DR region
aws route53 change-resource-record-sets \
  --hosted-zone-id $HOSTED_ZONE_ID \
  --change-batch file://dr-failover.json
```

#### 2.3 Communication
- **Internal**: Notify engineering teams via Slack
- **External**: Update status page and send customer notifications
- **Stakeholders**: Notify management and compliance teams

---

### Phase 3: Recovery (15-60 minutes)

#### Scenario A: API Service Failure

**Automated Recovery:**
```bash
# Trigger automated recovery
aws lambda invoke \
  --function-name healthcare-platform-recovery \
  --payload '{"scenario": "api_failure", "region": "us-east-1"}' \
  response.json
```

**Manual Recovery Steps:**
1. Scale down unhealthy tasks
2. Deploy from last known good image
3. Gradually increase traffic (5% → 25% → 50% → 100%)
4. Monitor error rates and latency

#### Scenario B: Database Failure

**Automated Recovery:**
```bash
# Promote read replica to master
aws rds promote-read-replica \
  --db-instance-identifier healthcare-platform-dr \
  --backup-retention-period 7

# Update connection strings
aws lambda invoke \
  --function-name update-database-endpoints \
  --payload '{"new_endpoint": "healthcare-platform-dr.cluster-xxxx.us-west-2.rds.amazonaws.com"}' \
  response.json
```

**Manual Recovery Steps:**
1. Promote DR database instance
2. Update application configuration
3. Verify data consistency
4. Resume write operations

#### Scenario C: Regional Failure

**Automated Recovery:**
```bash
# DNS failover to DR region
aws route53 change-resource-record-sets \
  --hosted-zone-id $HOSTED_ZONE_ID \
  --change-batch '{
    "Changes": [{
      "Action": "UPSERT",
      "ResourceRecordSet": {
        "Name": "api.healthcare-platform.com",
        "Type": "CNAME",
        "TTL": 60,
        "ResourceRecords": [{"Value": "dr-alb-xxxx.us-west-2.elb.amazonaws.com"}]
      }
    }]
  }'
```

**Manual Recovery Steps:**
1. Verify DR region capacity
2. Update DNS records
3. Scale up DR infrastructure
4. Redirect user traffic

#### Scenario D: Data Corruption

**Recovery Steps:**
1. Stop all write operations
2. Restore from last known good backup
3. Verify data integrity
4. Gradually resume operations
5. Investigate root cause

---

### Phase 4: Testing & Validation (60-120 minutes)

#### 4.1 Functional Testing
```bash
# Run automated tests
pytest tests/integration_tests.py -v
pytest tests/e2e_tests.py -v

# Manual verification checklist
- [ ] User authentication works
- [ ] Patient data retrieval functions
- [ ] Health metrics submission works
- [ ] Appointments can be scheduled
- [ ] Reports generate correctly
- [ ] API response times < 500ms
- [ ] Error rate < 1%
```

#### 4.2 Performance Validation
```bash
# Load testing
locust -f tests/load_tests.py \
  --host https://api.healthcare-platform.com \
  --users 100 \
  --spawn-rate 10 \
  --run-time 5m

# Performance benchmarks
ab -n 1000 -c 10 https://api.healthcare-platform.com/api/health/
```

#### 4.3 Data Integrity Checks
```bash
# Database consistency checks
psql -h $DB_HOST -d healthcare_platform -c "
  SELECT schemaname, tablename, n_tup_ins, n_tup_upd, n_tup_del
  FROM pg_stat_user_tables
  ORDER BY n_tup_ins DESC LIMIT 10;
"

# Data validation
python scripts/data_validation.py --check-integrity
```

---

### Phase 5: Return to Normal (2-24 hours)

#### 5.1 Failback Planning
- Assess primary region recovery
- Plan gradual traffic shift back
- Schedule maintenance window
- Prepare rollback procedures

#### 5.2 Gradual Recovery
```bash
# Gradual traffic shift back to primary
aws route53 change-resource-record-sets \
  --hosted-zone-id $HOSTED_ZONE_ID \
  --change-batch '{
    "Comment": "Gradual failback to primary region",
    "Changes": [{
      "Action": "UPSERT",
      "ResourceRecordSet": {
        "Name": "api.healthcare-platform.com",
        "Type": "CNAME",
        "SetIdentifier": "primary",
        "Weight": 30,
        "TTL": 60,
        "ResourceRecords": [{"Value": "primary-alb-xxxx.us-east-1.elb.amazonaws.com"}]
      }
    }, {
      "Action": "UPSERT",
      "ResourceRecordSet": {
        "Name": "api.healthcare-platform.com",
        "Type": "CNAME",
        "SetIdentifier": "dr",
        "Weight": 70,
        "TTL": 60,
        "ResourceRecords": [{"Value": "dr-alb-xxxx.us-west-2.elb.amazonaws.com"}]
      }
    }]
  }'
```

#### 5.3 Full Recovery Verification
- All automated tests pass
- Performance metrics within normal ranges
- Data consistency verified
- User feedback collected

---

## Component-Specific Recovery Procedures

### API Service Recovery

**Quick Recovery:**
```bash
# Force deployment of last known good version
aws ecs update-service \
  --cluster healthcare-platform-production \
  --service api-service \
  --force-new-deployment \
  --task-definition healthcare-platform-api:good-known-version
```

**Full Recovery:**
1. Identify root cause
2. Fix application code
3. Build and test new image
4. Deploy with canary strategy
5. Monitor and gradually increase traffic

### Database Recovery

**Point-in-Time Recovery:**
```bash
# Create new instance from backup
aws rds restore-db-instance-from-db-snapshot \
  --db-instance-identifier healthcare-platform-recovery \
  --db-snapshot-identifier healthcare-platform-backup-2024-01-15 \
  --db-instance-class db.r6g.large \
  --vpc-security-group-ids $SECURITY_GROUP_ID \
  --db-subnet-group-name healthcare-platform-db-subnet

# Point-in-time recovery
aws rds restore-db-instance-to-point-in-time \
  --source-db-instance-identifier healthcare-platform-prod \
  --target-db-instance-identifier healthcare-platform-pitr \
  --restore-time 2024-01-15T10:00:00Z
```

### Frontend Recovery

**CDN Failover:**
```bash
# Switch to static maintenance page
aws s3 cp s3://healthcare-platform-dr/frontend/index.html s3://healthcare-platform-frontend/index.html

# Update CloudFront distribution
aws cloudfront create-invalidation \
  --distribution-id $CLOUDFRONT_ID \
  --paths "/*"
```

---

## Monitoring & Alerting

### Key Metrics to Monitor

```yaml
# Application Metrics
api_response_time: < 500ms
error_rate: < 1%
availability: > 99.9%

# Infrastructure Metrics
cpu_utilization: < 80%
memory_utilization: < 85%
disk_usage: < 90%
network_errors: < 0.1%

# Business Metrics
active_users: > 1000
successful_transactions: > 95%
data_processing_time: < 30s
```

### Alert Configuration

```yaml
# Critical Alerts
- metric: api_error_rate > 5%
  channels: [pagerduty, slack, email]
  escalation: immediate

- metric: database_connection_timeout > 10
  channels: [pagerduty, slack]
  escalation: immediate

# Warning Alerts
- metric: api_response_time > 1000ms
  channels: [slack]
  escalation: 15 minutes

- metric: disk_usage > 85%
  channels: [slack, email]
  escalation: 30 minutes
```

---

## Prevention Measures

### Regular Testing
- **Weekly**: Component failover testing
- **Monthly**: Full disaster recovery simulation
- **Quarterly**: Cross-region failover testing
- **Annually**: Complete infrastructure rebuild testing

### Backup Verification
```bash
# Automated backup verification
#!/bin/bash
BACKUP_FILE=$(aws s3 ls s3://healthcare-platform-backups/ | tail -1 | awk '{print $4}')
aws s3 cp s3://healthcare-platform-backups/$BACKUP_FILE /tmp/
pg_restore -l /tmp/$BACKUP_FILE > /dev/null
if [ $? -eq 0 ]; then
  echo "Backup verification successful"
else
  echo "Backup verification failed"
  exit 1
fi
```

### Capacity Planning
- Monitor resource utilization trends
- Plan for 2x peak capacity
- Regular load testing
- Cost-benefit analysis of scaling decisions

---

## Lessons Learned & Improvements

### Post-Incident Review Process

1. **Timeline Reconstruction**
   - Map out all events leading to incident
   - Identify detection and response times
   - Document communication effectiveness

2. **Root Cause Analysis**
   - Technical root cause identification
   - Process and human factor analysis
   - Systemic weakness identification

3. **Action Items**
   - Immediate fixes implementation
   - Long-term improvement planning
   - Prevention measure deployment

4. **Documentation Updates**
   - Update runbook with new procedures
   - Add monitoring for previously unknown failure modes
   - Update contact lists and escalation procedures

### Continuous Improvement

- **Automated Testing**: Increase test coverage
- **Monitoring Enhancement**: Add more granular metrics
- **Process Documentation**: Keep runbook current
- **Team Training**: Regular disaster recovery drills
- **Technology Updates**: Stay current with best practices

---

## Appendices

### A. Contact Lists
### B. System Architecture Diagrams
### C. Backup Procedures
### D. Communication Templates
### E. Recovery Scripts
### F. Performance Baselines
### G. Compliance Requirements
