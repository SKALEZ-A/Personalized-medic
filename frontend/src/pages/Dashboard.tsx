import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
  Chip,
  LinearProgress,
  Avatar,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Alert,
  Snackbar,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  Notifications,
  MedicalServices,
  Science,
  Assessment,
  Timeline,
  Warning,
  CheckCircle,
  Schedule,
  Person,
} from '@mui/icons-material';
import { useAuth } from '../contexts/AuthContext';
import { useHealthData } from '../contexts/HealthDataContext';

// Types
interface HealthMetric {
  name: string;
  value: number;
  unit: string;
  status: 'normal' | 'warning' | 'critical';
  trend: 'up' | 'down' | 'stable';
  change: number;
}

interface RecentActivity {
  id: string;
  type: 'analysis' | 'appointment' | 'medication' | 'alert';
  title: string;
  description: string;
  timestamp: string;
  status: 'completed' | 'pending' | 'overdue';
}

interface UpcomingAppointment {
  id: string;
  title: string;
  doctor: string;
  date: string;
  time: string;
  type: string;
}

interface HealthAlert {
  id: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  title: string;
  message: string;
  timestamp: string;
  acknowledged: boolean;
}

const Dashboard: React.FC = () => {
  const { user } = useAuth();
  const { healthData, refreshHealthData } = useHealthData();

  const [healthMetrics, setHealthMetrics] = useState<HealthMetric[]>([]);
  const [recentActivities, setRecentActivities] = useState<RecentActivity[]>([]);
  const [upcomingAppointments, setUpcomingAppointments] = useState<UpcomingAppointment[]>([]);
  const [healthAlerts, setHealthAlerts] = useState<HealthAlert[]>([]);
  const [selectedAlert, setSelectedAlert] = useState<HealthAlert | null>(null);
  const [alertDialogOpen, setAlertDialogOpen] = useState(false);
  const [snackbar, setSnackbar] = useState({
    open: false,
    message: '',
    severity: 'success' as 'success' | 'error' | 'warning' | 'info',
  });

  // Load dashboard data
  useEffect(() => {
    loadDashboardData();
  }, [user]);

  const loadDashboardData = async () => {
    try {
      // Load health metrics
      const metrics: HealthMetric[] = [
        {
          name: 'Heart Rate',
          value: 72,
          unit: 'bpm',
          status: 'normal',
          trend: 'stable',
          change: 0,
        },
        {
          name: 'Blood Pressure',
          value: 120,
          unit: 'mmHg',
          status: 'normal',
          trend: 'down',
          change: -2,
        },
        {
          name: 'Blood Glucose',
          value: 95,
          unit: 'mg/dL',
          status: 'normal',
          trend: 'up',
          change: 1,
        },
        {
          name: 'BMI',
          value: 24.5,
          unit: '',
          status: 'normal',
          trend: 'stable',
          change: 0,
        },
      ];
      setHealthMetrics(metrics);

      // Load recent activities
      const activities: RecentActivity[] = [
        {
          id: '1',
          type: 'analysis',
          title: 'Genomic Analysis Completed',
          description: 'Your comprehensive genomic analysis is ready for review',
          timestamp: '2024-01-15T10:30:00Z',
          status: 'completed',
        },
        {
          id: '2',
          type: 'medication',
          title: 'Medication Reminder',
          description: 'Time to take your prescribed medication',
          timestamp: '2024-01-15T08:00:00Z',
          status: 'completed',
        },
        {
          id: '3',
          type: 'appointment',
          title: 'Cardiology Consultation',
          description: 'Scheduled appointment with Dr. Smith',
          timestamp: '2024-01-16T14:00:00Z',
          status: 'pending',
        },
      ];
      setRecentActivities(activities);

      // Load upcoming appointments
      const appointments: UpcomingAppointment[] = [
        {
          id: '1',
          title: 'Annual Physical',
          doctor: 'Dr. Sarah Johnson',
          date: '2024-01-20',
          time: '09:00',
          type: 'Physical Exam',
        },
        {
          id: '2',
          title: 'Cardiology Follow-up',
          doctor: 'Dr. Michael Chen',
          date: '2024-01-25',
          time: '11:30',
          type: 'Consultation',
        },
      ];
      setUpcomingAppointments(appointments);

      // Load health alerts
      const alerts: HealthAlert[] = [
        {
          id: '1',
          severity: 'medium',
          title: 'Elevated Blood Pressure',
          message: 'Your recent blood pressure reading was slightly elevated. Please monitor and consult your physician.',
          timestamp: '2024-01-14T16:45:00Z',
          acknowledged: false,
        },
        {
          id: '2',
          severity: 'low',
          title: 'Medication Refill Reminder',
          message: 'Your prescription for Metformin will expire in 7 days.',
          timestamp: '2024-01-13T09:15:00Z',
          acknowledged: true,
        },
      ];
      setHealthAlerts(alerts);

    } catch (error) {
      setSnackbar({
        open: true,
        message: 'Failed to load dashboard data',
        severity: 'error',
      });
    }
  };

  const handleAlertClick = (alert: HealthAlert) => {
    setSelectedAlert(alert);
    setAlertDialogOpen(true);
  };

  const handleAcknowledgeAlert = async () => {
    if (!selectedAlert) return;

    try {
      // API call to acknowledge alert
      // await api.post(`/alerts/${selectedAlert.id}/acknowledge`);

      setHealthAlerts(prev =>
        prev.map(alert =>
          alert.id === selectedAlert.id
            ? { ...alert, acknowledged: true }
            : alert
        )
      );

      setAlertDialogOpen(false);
      setSelectedAlert(null);

      setSnackbar({
        open: true,
        message: 'Alert acknowledged successfully',
        severity: 'success',
      });
    } catch (error) {
      setSnackbar({
        open: true,
        message: 'Failed to acknowledge alert',
        severity: 'error',
      });
    }
  };

  const getStatusColor = (status: HealthMetric['status']) => {
    switch (status) {
      case 'normal': return '#4caf50';
      case 'warning': return '#ff9800';
      case 'critical': return '#f44336';
      default: return '#9e9e9e';
    }
  };

  const getSeverityColor = (severity: HealthAlert['severity']) => {
    switch (severity) {
      case 'low': return '#2196f3';
      case 'medium': return '#ff9800';
      case 'high': return '#f44336';
      case 'critical': return '#d32f2f';
      default: return '#9e9e9e';
    }
  };

  const getActivityIcon = (type: RecentActivity['type']) => {
    switch (type) {
      case 'analysis': return <Science />;
      case 'appointment': return <Schedule />;
      case 'medication': return <MedicalServices />;
      case 'alert': return <Notifications />;
      default: return <Assessment />;
    }
  };

  const getActivityStatusColor = (status: RecentActivity['status']) => {
    switch (status) {
      case 'completed': return '#4caf50';
      case 'pending': return '#ff9800';
      case 'overdue': return '#f44336';
      default: return '#9e9e9e';
    }
  };

  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      {/* Header */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" gutterBottom>
          Welcome back, {user?.firstName}!
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Here's your personalized health overview for today.
        </Typography>
      </Box>

      {/* Health Metrics Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        {healthMetrics.map((metric, index) => (
          <Grid item xs={12} sm={6} md={3} key={index}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <Typography variant="h6" sx={{ flexGrow: 1 }}>
                    {metric.name}
                  </Typography>
                  <Box
                    sx={{
                      width: 12,
                      height: 12,
                      borderRadius: '50%',
                      backgroundColor: getStatusColor(metric.status),
                    }}
                  />
                </Box>
                <Typography variant="h4" sx={{ mb: 1 }}>
                  {metric.value} {metric.unit}
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  {metric.trend === 'up' && <TrendingUp color="error" fontSize="small" />}
                  {metric.trend === 'down' && <TrendingDown color="success" fontSize="small" />}
                  <Typography
                    variant="body2"
                    color={metric.trend === 'up' ? 'error' : 'success'}
                    sx={{ ml: 0.5 }}
                  >
                    {metric.change > 0 ? '+' : ''}{metric.change} from last week
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Main Content Grid */}
      <Grid container spacing={3}>
        {/* Recent Activities */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Recent Activities
              </Typography>
              <Box sx={{ maxHeight: 400, overflow: 'auto' }}>
                {recentActivities.map((activity, index) => (
                  <Box
                    key={activity.id}
                    sx={{
                      display: 'flex',
                      alignItems: 'center',
                      p: 2,
                      borderBottom: index < recentActivities.length - 1 ? '1px solid #e0e0e0' : 'none',
                    }}
                  >
                    <Avatar sx={{ mr: 2, bgcolor: 'primary.main' }}>
                      {getActivityIcon(activity.type)}
                    </Avatar>
                    <Box sx={{ flexGrow: 1 }}>
                      <Typography variant="subtitle2">
                        {activity.title}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        {activity.description}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        {new Date(activity.timestamp).toLocaleString()}
                      </Typography>
                    </Box>
                    <Chip
                      label={activity.status}
                      size="small"
                      sx={{
                        bgcolor: getActivityStatusColor(activity.status),
                        color: 'white',
                        textTransform: 'capitalize',
                      }}
                    />
                  </Box>
                ))}
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Upcoming Appointments */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Upcoming Appointments
              </Typography>
              <Box sx={{ maxHeight: 400, overflow: 'auto' }}>
                {upcomingAppointments.map((appointment, index) => (
                  <Box
                    key={appointment.id}
                    sx={{
                      display: 'flex',
                      alignItems: 'center',
                      p: 2,
                      borderBottom: index < upcomingAppointments.length - 1 ? '1px solid #e0e0e0' : 'none',
                    }}
                  >
                    <Avatar sx={{ mr: 2, bgcolor: 'secondary.main' }}>
                      <Person />
                    </Avatar>
                    <Box sx={{ flexGrow: 1 }}>
                      <Typography variant="subtitle2">
                        {appointment.title}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        {appointment.doctor}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        {new Date(appointment.date).toLocaleDateString()} at {appointment.time}
                      </Typography>
                    </Box>
                    <Chip label={appointment.type} size="small" variant="outlined" />
                  </Box>
                ))}
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Health Alerts */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Health Alerts
              </Typography>
              <Box sx={{ maxHeight: 300, overflow: 'auto' }}>
                {healthAlerts.map((alert, index) => (
                  <Box
                    key={alert.id}
                    sx={{
                      display: 'flex',
                      alignItems: 'center',
                      p: 2,
                      borderBottom: index < healthAlerts.length - 1 ? '1px solid #e0e0e0' : 'none',
                      cursor: 'pointer',
                      '&:hover': { bgcolor: 'action.hover' },
                    }}
                    onClick={() => handleAlertClick(alert)}
                  >
                    <Box sx={{ mr: 2 }}>
                      {alert.acknowledged ? (
                        <CheckCircle color="success" />
                      ) : (
                        <Warning sx={{ color: getSeverityColor(alert.severity) }} />
                      )}
                    </Box>
                    <Box sx={{ flexGrow: 1 }}>
                      <Typography variant="subtitle2">
                        {alert.title}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        {alert.message}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        {new Date(alert.timestamp).toLocaleString()}
                      </Typography>
                    </Box>
                    <Chip
                      label={alert.severity}
                      size="small"
                      sx={{
                        bgcolor: getSeverityColor(alert.severity),
                        color: 'white',
                        textTransform: 'capitalize',
                      }}
                    />
                  </Box>
                ))}
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Alert Dialog */}
      <Dialog open={alertDialogOpen} onClose={() => setAlertDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>
          {selectedAlert?.title}
        </DialogTitle>
        <DialogContent>
          <Typography variant="body1" sx={{ mb: 2 }}>
            {selectedAlert?.message}
          </Typography>
          <Typography variant="caption" color="text.secondary">
            Alert Time: {selectedAlert ? new Date(selectedAlert.timestamp).toLocaleString() : ''}
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setAlertDialogOpen(false)}>Close</Button>
          {!selectedAlert?.acknowledged && (
            <Button onClick={handleAcknowledgeAlert} variant="contained" color="primary">
              Acknowledge
            </Button>
          )}
        </DialogActions>
      </Dialog>

      {/* Snackbar for notifications */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={() => setSnackbar(prev => ({ ...prev, open: false }))}
      >
        <Alert
          onClose={() => setSnackbar(prev => ({ ...prev, open: false }))}
          severity={snackbar.severity}
          sx={{ width: '100%' }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default Dashboard;
