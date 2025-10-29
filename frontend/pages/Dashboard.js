import React, { useState, useEffect } from 'react';
import {
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  LinearProgress,
  Chip,
  Avatar,
  List,
  ListItem,
  ListItemAvatar,
  ListItemText,
  Divider,
  Button,
  Alert,
} from '@mui/material';
import {
  TrendingUp as TrendingUpIcon,
  People as PeopleIcon,
  Biotech as BiotechIcon,
  HealthAndSafety as HealthIcon,
  Assessment as AssessmentIcon,
  Notifications as NotificationsIcon,
  Schedule as ScheduleIcon,
  Warning as WarningIcon,
} from '@mui/icons-material';
import { useAuth } from '../hooks/useAuth';
import { useNotifications } from '../hooks/useNotifications';

const Dashboard = () => {
  const { user } = useAuth();
  const { notifications } = useNotifications();
  const [dashboardData, setDashboardData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Simulate loading dashboard data
    const loadDashboardData = async () => {
      // In a real app, this would fetch from the API
      const mockData = {
        stats: {
          totalPatients: 15420,
          activeAnalyses: 47,
          completedTrials: 23,
          systemHealth: 98.5,
        },
        recentActivity: [
          {
            id: 1,
            type: 'genomic_analysis',
            title: 'Genomic Analysis Completed',
            description: 'Analysis for patient PAT-001 completed successfully',
            timestamp: new Date(Date.now() - 1000 * 60 * 30), // 30 minutes ago
            icon: <BiotechIcon />,
          },
          {
            id: 2,
            type: 'clinical_trial',
            title: 'New Trial Enrollment',
            description: 'Patient enrolled in Diabetes Prevention Trial',
            timestamp: new Date(Date.now() - 1000 * 60 * 60 * 2), // 2 hours ago
            icon: <PeopleIcon />,
          },
          {
            id: 3,
            type: 'treatment_plan',
            title: 'Treatment Plan Updated',
            description: 'Optimized treatment plan for cardiovascular disease',
            timestamp: new Date(Date.now() - 1000 * 60 * 60 * 4), // 4 hours ago
            icon: <HealthIcon />,
          },
        ],
        alerts: [
          {
            id: 1,
            severity: 'warning',
            title: 'High System Load',
            message: 'Current analysis queue is above normal levels',
            timestamp: new Date(Date.now() - 1000 * 60 * 15),
          },
          {
            id: 2,
            severity: 'info',
            title: 'New Feature Available',
            message: 'Advanced genomic analysis now supports CNV detection',
            timestamp: new Date(Date.now() - 1000 * 60 * 60),
          },
        ],
        upcomingTasks: [
          {
            id: 1,
            title: 'Review Clinical Trial Results',
            description: 'Phase II diabetes trial interim analysis due',
            dueDate: new Date(Date.now() + 1000 * 60 * 60 * 24), // Tomorrow
            priority: 'high',
          },
          {
            id: 2,
            title: 'System Maintenance',
            description: 'Scheduled maintenance window',
            dueDate: new Date(Date.now() + 1000 * 60 * 60 * 24 * 3), // 3 days
            priority: 'medium',
          },
        ],
      };

      setTimeout(() => {
        setDashboardData(mockData);
        setLoading(false);
      }, 1000);
    };

    loadDashboardData();
  }, []);

  const formatTimeAgo = (timestamp) => {
    const now = new Date();
    const diff = now - timestamp;
    const minutes = Math.floor(diff / (1000 * 60));
    const hours = Math.floor(diff / (1000 * 60 * 60));
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));

    if (days > 0) return `${days}d ago`;
    if (hours > 0) return `${hours}h ago`;
    if (minutes > 0) return `${minutes}m ago`;
    return 'Just now';
  };

  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'error':
        return 'error';
      case 'warning':
        return 'warning';
      case 'info':
        return 'info';
      default:
        return 'default';
    }
  };

  const getPriorityColor = (priority) => {
    switch (priority) {
      case 'high':
        return 'error';
      case 'medium':
        return 'warning';
      case 'low':
        return 'success';
      default:
        return 'default';
    }
  };

  if (loading) {
    return (
      <Box sx={{ p: 3 }}>
        <LinearProgress />
        <Typography sx={{ mt: 2 }}>Loading dashboard...</Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Welcome back, {user?.name || 'User'}!
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
        Here's an overview of your AI Personalized Medicine Platform
      </Typography>

      {/* Stats Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <PeopleIcon color="primary" sx={{ mr: 1 }} />
                <Typography variant="h6">Total Patients</Typography>
              </Box>
              <Typography variant="h4" color="primary">
                {dashboardData?.stats?.totalPatients?.toLocaleString() || '0'}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                +12% from last month
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <BiotechIcon color="secondary" sx={{ mr: 1 }} />
                <Typography variant="h6">Active Analyses</Typography>
              </Box>
              <Typography variant="h4" color="secondary">
                {dashboardData?.stats?.activeAnalyses || 0}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Processing genomic data
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <AssessmentIcon color="success" sx={{ mr: 1 }} />
                <Typography variant="h6">Clinical Trials</Typography>
              </Box>
              <Typography variant="h4" sx={{ color: 'success.main' }}>
                {dashboardData?.stats?.completedTrials || 0}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Active studies
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <HealthIcon color="warning" sx={{ mr: 1 }} />
                <Typography variant="h6">System Health</Typography>
              </Box>
              <Typography variant="h4" sx={{ color: 'warning.main' }}>
                {dashboardData?.stats?.systemHealth || 0}%
              </Typography>
              <Typography variant="body2" color="text.secondary">
                All systems operational
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Main Content Grid */}
      <Grid container spacing={3}>
        {/* Recent Activity */}
        <Grid item xs={12} lg={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                <TrendingUpIcon sx={{ mr: 1 }} />
                Recent Activity
              </Typography>
              <List>
                {dashboardData?.recentActivity?.map((activity, index) => (
                  <React.Fragment key={activity.id}>
                    <ListItem alignItems="flex-start">
                      <ListItemAvatar>
                        <Avatar sx={{ bgcolor: 'primary.main' }}>
                          {activity.icon}
                        </Avatar>
                      </ListItemAvatar>
                      <ListItemText
                        primary={activity.title}
                        secondary={
                          <>
                            <Typography variant="body2" color="text.secondary">
                              {activity.description}
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                              {formatTimeAgo(activity.timestamp)}
                            </Typography>
                          </>
                        }
                      />
                    </ListItem>
                    {index < dashboardData.recentActivity.length - 1 && <Divider />}
                  </React.Fragment>
                ))}
              </List>
            </CardContent>
          </Card>
        </Grid>

        {/* Alerts and Notifications */}
        <Grid item xs={12} lg={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                <NotificationsIcon sx={{ mr: 1 }} />
                System Alerts
              </Typography>
              <Box sx={{ mb: 2 }}>
                {dashboardData?.alerts?.map((alert) => (
                  <Alert
                    key={alert.id}
                    severity={getSeverityColor(alert.severity)}
                    sx={{ mb: 1 }}
                    icon={<WarningIcon />}
                  >
                    <Typography variant="body2" fontWeight="medium">
                      {alert.title}
                    </Typography>
                    <Typography variant="body2">
                      {alert.message}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {formatTimeAgo(alert.timestamp)}
                    </Typography>
                  </Alert>
                ))}
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Upcoming Tasks */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                <ScheduleIcon sx={{ mr: 1 }} />
                Upcoming Tasks
              </Typography>
              <Grid container spacing={2}>
                {dashboardData?.upcomingTasks?.map((task) => (
                  <Grid item xs={12} md={6} key={task.id}>
                    <Card variant="outlined" sx={{ p: 2 }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1 }}>
                        <Typography variant="h6" component="div">
                          {task.title}
                        </Typography>
                        <Chip
                          label={task.priority}
                          size="small"
                          color={getPriorityColor(task.priority)}
                          variant="outlined"
                        />
                      </Box>
                      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                        {task.description}
                      </Typography>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <Typography variant="caption" color="text.secondary">
                          Due: {task.dueDate.toLocaleDateString()}
                        </Typography>
                        <Button size="small" variant="outlined">
                          View Details
                        </Button>
                      </Box>
                    </Card>
                  </Grid>
                ))}
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard;
