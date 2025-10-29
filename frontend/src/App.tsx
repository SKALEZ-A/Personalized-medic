import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { CssBaseline, Box, CircularProgress } from '@mui/material';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import { NotificationProvider } from './contexts/NotificationContext';
import { HealthDataProvider } from './contexts/HealthDataContext';

// Layout Components
import Header from './components/layout/Header';
import Sidebar from './components/layout/Sidebar';
import Footer from './components/layout/Footer';

// Page Components
import Dashboard from './pages/Dashboard';
import PatientProfile from './pages/PatientProfile';
import GenomicAnalysis from './pages/GenomicAnalysis';
import HealthMonitoring from './pages/HealthMonitoring';
import TreatmentPlans from './pages/TreatmentPlans';
import ClinicalSupport from './pages/ClinicalSupport';
import DrugDiscovery from './pages/DrugDiscovery';
import ResearchTrials from './pages/ResearchTrials';
import VirtualAssistant from './pages/VirtualAssistant';
import Reports from './pages/Reports';
import Settings from './pages/Settings';

// Auth Components
import Login from './pages/auth/Login';
import Register from './pages/auth/Register';
import MFAVerification from './pages/auth/MFAVerification';
import ForgotPassword from './pages/auth/ForgotPassword';

// Admin Components
import AdminDashboard from './pages/admin/AdminDashboard';
import UserManagement from './pages/admin/UserManagement';
import SystemMetrics from './pages/admin/SystemMetrics';
import AuditLogs from './pages/admin/AuditLogs';

// Error Components
import NotFound from './pages/NotFound';
import ServerError from './pages/ServerError';
import Unauthorized from './pages/Unauthorized';

// Theme configuration
const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
      light: '#42a5f5',
      dark: '#1565c0',
    },
    secondary: {
      main: '#dc004e',
      light: '#ff5983',
      dark: '#9a0036',
    },
    background: {
      default: '#f5f5f5',
      paper: '#ffffff',
    },
    text: {
      primary: '#212121',
      secondary: '#757575',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontSize: '2.5rem',
      fontWeight: 600,
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 600,
    },
    h3: {
      fontSize: '1.75rem',
      fontWeight: 600,
    },
    h4: {
      fontSize: '1.5rem',
      fontWeight: 600,
    },
    h5: {
      fontSize: '1.25rem',
      fontWeight: 600,
    },
    h6: {
      fontSize: '1rem',
      fontWeight: 600,
    },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          textTransform: 'none',
          fontWeight: 600,
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          boxShadow: '0 4px 6px rgba(0, 0, 0, 0.07)',
        },
      },
    },
  },
});

// Protected Route Component
const ProtectedRoute: React.FC<{ children: React.ReactNode; requiredRole?: string }> = ({
  children,
  requiredRole
}) => {
  const { isAuthenticated, user, loading } = useAuth();

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="100vh">
        <CircularProgress />
      </Box>
    );
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  if (requiredRole && user?.role !== requiredRole && user?.role !== 'admin') {
    return <Navigate to="/unauthorized" replace />;
  }

  return <>{children}</>;
};

// Main Layout Component
const MainLayout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen);
  };

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh' }}>
      <Header onMenuClick={toggleSidebar} />
      <Sidebar open={sidebarOpen} onClose={() => setSidebarOpen(false)} />
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: 3,
          width: { sm: `calc(100% - ${sidebarOpen ? '240px' : '0px'})` },
          ml: { sm: sidebarOpen ? '240px' : 0 },
          transition: 'margin-left 0.3s ease, width 0.3s ease',
        }}
      >
        {children}
      </Box>
      <Footer />
    </Box>
  );
};

// App Content Component
const AppContent: React.FC = () => {
  const { isAuthenticated, loading } = useAuth();

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="100vh">
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Router>
      <Routes>
        {/* Public Routes */}
        <Route path="/login" element={
          isAuthenticated ? <Navigate to="/dashboard" replace /> : <Login />
        } />
        <Route path="/register" element={
          isAuthenticated ? <Navigate to="/dashboard" replace /> : <Register />
        } />
        <Route path="/forgot-password" element={<ForgotPassword />} />
        <Route path="/mfa-verification" element={<MFAVerification />} />

        {/* Protected Routes */}
        <Route path="/" element={
          <ProtectedRoute>
            <MainLayout>
              <Navigate to="/dashboard" replace />
            </MainLayout>
          </ProtectedRoute>
        } />

        <Route path="/dashboard" element={
          <ProtectedRoute>
            <MainLayout>
              <Dashboard />
            </MainLayout>
          </ProtectedRoute>
        } />

        <Route path="/profile" element={
          <ProtectedRoute>
            <MainLayout>
              <PatientProfile />
            </MainLayout>
          </ProtectedRoute>
        } />

        <Route path="/genomic-analysis" element={
          <ProtectedRoute>
            <MainLayout>
              <GenomicAnalysis />
            </MainLayout>
          </ProtectedRoute>
        } />

        <Route path="/health-monitoring" element={
          <ProtectedRoute>
            <MainLayout>
              <HealthMonitoring />
            </MainLayout>
          </ProtectedRoute>
        } />

        <Route path="/treatment-plans" element={
          <ProtectedRoute>
            <MainLayout>
              <TreatmentPlans />
            </MainLayout>
          </ProtectedRoute>
        } />

        <Route path="/clinical-support" element={
          <ProtectedRoute>
            <MainLayout>
              <ClinicalSupport />
            </MainLayout>
          </ProtectedRoute>
        } />

        <Route path="/drug-discovery" element={
          <ProtectedRoute requiredRole="researcher">
            <MainLayout>
              <DrugDiscovery />
            </MainLayout>
          </ProtectedRoute>
        } />

        <Route path="/research-trials" element={
          <ProtectedRoute requiredRole="researcher">
            <MainLayout>
              <ResearchTrials />
            </MainLayout>
          </ProtectedRoute>
        } />

        <Route path="/virtual-assistant" element={
          <ProtectedRoute>
            <MainLayout>
              <VirtualAssistant />
            </MainLayout>
          </ProtectedRoute>
        } />

        <Route path="/reports" element={
          <ProtectedRoute>
            <MainLayout>
              <Reports />
            </MainLayout>
          </ProtectedRoute>
        } />

        <Route path="/settings" element={
          <ProtectedRoute>
            <MainLayout>
              <Settings />
            </MainLayout>
          </ProtectedRoute>
        } />

        {/* Admin Routes */}
        <Route path="/admin" element={
          <ProtectedRoute requiredRole="admin">
            <MainLayout>
              <AdminDashboard />
            </MainLayout>
          </ProtectedRoute>
        } />

        <Route path="/admin/users" element={
          <ProtectedRoute requiredRole="admin">
            <MainLayout>
              <UserManagement />
            </MainLayout>
          </ProtectedRoute>
        } />

        <Route path="/admin/metrics" element={
          <ProtectedRoute requiredRole="admin">
            <MainLayout>
              <SystemMetrics />
            </MainLayout>
          </ProtectedRoute>
        } />

        <Route path="/admin/audit" element={
          <ProtectedRoute requiredRole="admin">
            <MainLayout>
              <AuditLogs />
            </MainLayout>
          </ProtectedRoute>
        } />

        {/* Error Routes */}
        <Route path="/unauthorized" element={<Unauthorized />} />
        <Route path="/server-error" element={<ServerError />} />
        <Route path="*" element={<NotFound />} />
      </Routes>
    </Router>
  );
};

// Main App Component
const App: React.FC = () => {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <AuthProvider>
        <NotificationProvider>
          <HealthDataProvider>
            <AppContent />
          </HealthDataProvider>
        </NotificationProvider>
      </AuthProvider>
    </ThemeProvider>
  );
};

export default App;
