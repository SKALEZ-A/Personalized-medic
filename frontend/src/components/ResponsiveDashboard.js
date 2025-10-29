/**
 * Responsive Dashboard Component for AI Personalized Medicine Platform
 * Mobile-first design with PWA capabilities and accessibility features
 */

import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useMediaQuery } from 'react-responsive';
import './ResponsiveDashboard.css';

const ResponsiveDashboard = ({ user, healthData, notifications }) => {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [activeTab, setActiveTab] = useState('overview');
  const [searchQuery, setSearchQuery] = useState('');
  const [theme, setTheme] = useState('light');
  const [offlineStatus, setOfflineStatus] = useState(!navigator.onLine);

  const isMobile = useMediaQuery({ maxWidth: 768 });
  const isTablet = useMediaQuery({ minWidth: 769, maxWidth: 1024 });
  const isDesktop = useMediaQuery({ minWidth: 1025 });

  // PWA and accessibility refs
  const mainContentRef = useRef(null);
  const sidebarRef = useRef(null);

  // PWA offline detection
  useEffect(() => {
    const handleOnline = () => setOfflineStatus(false);
    const handleOffline = () => setOfflineStatus(true);

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  // Theme management
  useEffect(() => {
    const savedTheme = localStorage.getItem('dashboard-theme') || 'light';
    setTheme(savedTheme);
    document.documentElement.setAttribute('data-theme', savedTheme);
  }, []);

  const toggleTheme = () => {
    const newTheme = theme === 'light' ? 'dark' : 'light';
    setTheme(newTheme);
    localStorage.setItem('dashboard-theme', newTheme);
    document.documentElement.setAttribute('data-theme', newTheme);
  };

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') {
        setSidebarOpen(false);
      }
      if (e.key === '/' && e.ctrlKey) {
        e.preventDefault();
        document.getElementById('search-input')?.focus();
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, []);

  // Skip to main content
  const skipToMain = () => {
    mainContentRef.current?.focus();
    mainContentRef.current?.scrollIntoView();
  };

  const navigationItems = [
    { id: 'overview', label: 'Overview', icon: '📊' },
    { id: 'health', label: 'Health Metrics', icon: '❤️' },
    { id: 'appointments', label: 'Appointments', icon: '📅' },
    { id: 'medications', label: 'Medications', icon: '💊' },
    { id: 'reports', label: 'Reports', icon: '📋' },
    { id: 'settings', label: 'Settings', icon: '⚙️' }
  ];

  const renderSidebar = () => (
    <motion.aside
      ref={sidebarRef}
      className={`sidebar ${sidebarOpen ? 'open' : ''}`}
      initial={false}
      animate={{ x: sidebarOpen ? 0 : (isMobile ? -280 : -320) }}
      transition={{ type: 'spring', stiffness: 300, damping: 30 }}
      role="navigation"
      aria-label="Main navigation"
    >
      <div className="sidebar-header">
        <div className="logo">
          <span className="logo-icon">🏥</span>
          <span className="logo-text">MediCare AI</span>
        </div>
        <button
          className="close-sidebar-btn"
          onClick={() => setSidebarOpen(false)}
          aria-label="Close navigation menu"
        >
          ✕
        </button>
      </div>

      <nav className="sidebar-nav">
        <ul role="menubar">
          {navigationItems.map((item) => (
            <li key={item.id} role="none">
              <button
                role="menuitem"
                className={`nav-item ${activeTab === item.id ? 'active' : ''}`}
                onClick={() => {
                  setActiveTab(item.id);
                  if (isMobile) setSidebarOpen(false);
                }}
                aria-current={activeTab === item.id ? 'page' : undefined}
              >
                <span className="nav-icon" aria-hidden="true">{item.icon}</span>
                <span className="nav-label">{item.label}</span>
              </button>
            </li>
          ))}
        </ul>
      </nav>

      <div className="sidebar-footer">
        <div className="user-info">
          <div className="user-avatar">
            {user?.name?.charAt(0)?.toUpperCase() || 'U'}
          </div>
          <div className="user-details">
            <div className="user-name">{user?.name || 'User'}</div>
            <div className="user-role">{user?.role || 'Patient'}</div>
          </div>
        </div>
      </div>
    </motion.aside>
  );

  const renderHeader = () => (
    <header className="dashboard-header" role="banner">
      <div className="header-left">
        <button
          className="menu-toggle"
          onClick={() => setSidebarOpen(!sidebarOpen)}
          aria-label="Toggle navigation menu"
          aria-expanded={sidebarOpen}
        >
          <span className="hamburger-icon">☰</span>
        </button>

        <div className="search-container">
          <input
            id="search-input"
            type="search"
            placeholder="Search health records, appointments..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="search-input"
            aria-label="Search health records"
          />
          <button className="search-btn" aria-label="Search">
            🔍
          </button>
        </div>
      </div>

      <div className="header-right">
        <button
          className="theme-toggle"
          onClick={toggleTheme}
          aria-label={`Switch to ${theme === 'light' ? 'dark' : 'light'} theme`}
        >
          {theme === 'light' ? '🌙' : '☀️'}
        </button>

        <button
          className="notifications-btn"
          aria-label={`Notifications ${notifications?.length ? `(${notifications.length} unread)` : ''}`}
        >
          🔔
          {notifications?.length > 0 && (
            <span className="notification-badge" aria-hidden="true">
              {notifications.length}
            </span>
          )}
        </button>

        <div className={`connection-status ${offlineStatus ? 'offline' : 'online'}`}>
          <span className="status-icon" aria-hidden="true">
            {offlineStatus ? '📶' : '📱'}
          </span>
          <span className="status-text sr-only">
            {offlineStatus ? 'Offline' : 'Online'}
          </span>
        </div>
      </div>
    </header>
  );

  const renderMainContent = () => {
    const tabContent = {
      overview: <OverviewTab healthData={healthData} />,
      health: <HealthTab healthData={healthData} />,
      appointments: <AppointmentsTab />,
      medications: <MedicationsTab />,
      reports: <ReportsTab />,
      settings: <SettingsTab user={user} theme={theme} onThemeChange={toggleTheme} />
    };

    return (
      <main
        ref={mainContentRef}
        className="main-content"
        tabIndex="-1"
        role="main"
        aria-label="Main content"
      >
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
          >
            {tabContent[activeTab]}
          </motion.div>
        </AnimatePresence>
      </main>
    );
  };

  return (
    <div className={`dashboard-container ${theme}-theme`}>
      {/* Skip to main content link for accessibility */}
      <a
        href="#main-content"
        className="skip-link sr-only"
        onClick={skipToMain}
      >
        Skip to main content
      </a>

      {/* Offline notification */}
      <AnimatePresence>
        {offlineStatus && (
          <motion.div
            className="offline-banner"
            initial={{ y: -50, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            exit={{ y: -50, opacity: 0 }}
            role="alert"
            aria-live="assertive"
          >
            <span className="offline-icon">📶</span>
            <span>You are currently offline. Some features may be limited.</span>
          </motion.div>
        )}
      </AnimatePresence>

      {renderSidebar()}

      {/* Mobile overlay */}
      {isMobile && sidebarOpen && (
        <div
          className="sidebar-overlay"
          onClick={() => setSidebarOpen(false)}
          aria-hidden="true"
        />
      )}

      <div className="dashboard-main">
        {renderHeader()}
        {renderMainContent()}
      </div>
    </div>
  );
};

// Overview Tab Component
const OverviewTab = ({ healthData }) => (
  <div className="tab-content">
    <h1>Health Overview</h1>

    <div className="metrics-grid">
      <motion.div
        className="metric-card"
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
      >
        <div className="metric-icon">❤️</div>
        <div className="metric-content">
          <h3>Heart Rate</h3>
          <div className="metric-value">
            {healthData?.heartRate || 72} <span className="metric-unit">BPM</span>
          </div>
          <div className="metric-trend positive">↗️ +2%</div>
        </div>
      </motion.div>

      <motion.div
        className="metric-card"
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
      >
        <div className="metric-icon">🩸</div>
        <div className="metric-content">
          <h3>Blood Pressure</h3>
          <div className="metric-value">
            {healthData?.bloodPressure || '120/80'}
          </div>
          <div className="metric-trend neutral">→ 0%</div>
        </div>
      </motion.div>

      <motion.div
        className="metric-card"
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
      >
        <div className="metric-icon">🏃‍♂️</div>
        <div className="metric-content">
          <h3>Steps Today</h3>
          <div className="metric-value">
            {healthData?.steps || 8432}
          </div>
          <div className="metric-trend positive">↗️ +15%</div>
        </div>
      </motion.div>

      <motion.div
        className="metric-card"
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
      >
        <div className="metric-icon">😴</div>
        <div className="metric-content">
          <h3>Sleep Hours</h3>
          <div className="metric-value">
            {healthData?.sleepHours || 7.5} <span className="metric-unit">hrs</span>
          </div>
          <div className="metric-trend negative">↘️ -5%</div>
        </div>
      </motion.div>
    </div>

    <div className="recent-activity">
      <h2>Recent Activity</h2>
      <div className="activity-list">
        <div className="activity-item">
          <div className="activity-icon">📅</div>
          <div className="activity-content">
            <div className="activity-title">Doctor Appointment</div>
            <div className="activity-time">Tomorrow at 2:00 PM</div>
          </div>
        </div>
        <div className="activity-item">
          <div className="activity-icon">💊</div>
          <div className="activity-content">
            <div className="activity-title">Medication Reminder</div>
            <div className="activity-time">Take vitamins in 2 hours</div>
          </div>
        </div>
      </div>
    </div>
  </div>
);

// Health Tab Component
const HealthTab = ({ healthData }) => (
  <div className="tab-content">
    <h1>Health Metrics</h1>

    <div className="health-charts">
      <div className="chart-container">
        <h3>Heart Rate Trend</h3>
        <div className="chart-placeholder">
          <div className="chart-line" style={{width: '70%'}}></div>
          <div className="chart-labels">
            <span>60</span><span>80</span><span>100</span><span>120</span>
          </div>
        </div>
      </div>

      <div className="chart-container">
        <h3>Blood Pressure History</h3>
        <div className="chart-placeholder">
          <div className="chart-bar" style={{height: '80%'}}></div>
          <div className="chart-bar" style={{height: '75%'}}></div>
          <div className="chart-bar" style={{height: '85%'}}></div>
        </div>
      </div>
    </div>

    <div className="health-insights">
      <h2>AI Health Insights</h2>
      <div className="insight-cards">
        <div className="insight-card positive">
          <div className="insight-icon">✅</div>
          <div className="insight-content">
            <h4>Excellent Recovery</h4>
            <p>Your heart rate variability indicates good recovery from recent activity.</p>
          </div>
        </div>

        <div className="insight-card warning">
          <div className="insight-icon">⚠️</div>
          <div className="insight-content">
            <h4>Sleep Quality</h4>
            <p>Consider improving your sleep schedule. Aim for 7-9 hours nightly.</p>
          </div>
        </div>
      </div>
    </div>
  </div>
);

// Placeholder components for other tabs
const AppointmentsTab = () => (
  <div className="tab-content">
    <h1>Appointments</h1>
    <p>Appointment management interface would go here.</p>
  </div>
);

const MedicationsTab = () => (
  <div className="tab-content">
    <h1>Medications</h1>
    <p>Medication tracking interface would go here.</p>
  </div>
);

const ReportsTab = () => (
  <div className="tab-content">
    <h1>Reports</h1>
    <p>Medical reports and analytics would go here.</p>
  </div>
);

const SettingsTab = ({ user, theme, onThemeChange }) => (
  <div className="tab-content">
    <h1>Settings</h1>

    <div className="settings-section">
      <h2>Appearance</h2>
      <div className="setting-item">
        <label htmlFor="theme-toggle">Theme</label>
        <button
          id="theme-toggle"
          className="theme-toggle-btn"
          onClick={onThemeChange}
        >
          {theme === 'light' ? '🌙 Dark Mode' : '☀️ Light Mode'}
        </button>
      </div>
    </div>

    <div className="settings-section">
      <h2>Accessibility</h2>
      <div className="setting-item">
        <label htmlFor="high-contrast">High Contrast</label>
        <input type="checkbox" id="high-contrast" />
      </div>
      <div className="setting-item">
        <label htmlFor="large-text">Large Text</label>
        <input type="checkbox" id="large-text" />
      </div>
    </div>

    <div className="settings-section">
      <h2>Notifications</h2>
      <div className="setting-item">
        <label htmlFor="email-notifications">Email Notifications</label>
        <input type="checkbox" id="email-notifications" defaultChecked />
      </div>
      <div className="setting-item">
        <label htmlFor="push-notifications">Push Notifications</label>
        <input type="checkbox" id="push-notifications" defaultChecked />
      </div>
    </div>
  </div>
);

export default ResponsiveDashboard;
