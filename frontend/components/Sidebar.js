import React from 'react';
import {
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Divider,
  Box,
  Typography,
  Chip,
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  Person as PersonIcon,
  Biotech as BiotechIcon,
  Science as ScienceIcon,
  HealthAndSafety as HealthIcon,
  MonitorHeart as MonitorIcon,
  Healing as HealingIcon,
  VideoCall as VideoIcon,
  Assessment as AssessmentIcon,
  AdminPanelSettings as AdminIcon,
  Analytics as AnalyticsIcon,
  Vaccines as VaccinesIcon,
} from '@mui/icons-material';
import { useNavigate, useLocation } from 'react-router-dom';

const drawerWidth = 280;

const Sidebar = ({ open, onToggle, userRole }) => {
  const navigate = useNavigate();
  const location = useLocation();

  const menuItems = [
    {
      text: 'Dashboard',
      icon: <DashboardIcon />,
      path: '/dashboard',
      roles: ['admin', 'doctor', 'patient'],
    },
    {
      text: 'Patient Portal',
      icon: <PersonIcon />,
      path: '/patient',
      roles: ['patient'],
    },
    {
      text: 'Genomic Analysis',
      icon: <BiotechIcon />,
      path: '/genomics',
      roles: ['admin', 'doctor'],
      badge: 'AI',
    },
    {
      text: 'Drug Discovery',
      icon: <ScienceIcon />,
      path: '/drug-discovery',
      roles: ['admin', 'researcher'],
      badge: 'AI',
    },
    {
      text: 'Clinical Trials',
      icon: <VaccinesIcon />,
      path: '/clinical-trials',
      roles: ['admin', 'doctor', 'researcher'],
    },
    {
      text: 'Health Monitoring',
      icon: <MonitorIcon />,
      path: '/monitoring',
      roles: ['admin', 'doctor', 'patient'],
      badge: 'IoT',
    },
    {
      text: 'Treatment Planning',
      icon: <HealingIcon />,
      path: '/treatment',
      roles: ['admin', 'doctor'],
      badge: 'AI',
    },
    {
      text: 'Telemedicine',
      icon: <VideoIcon />,
      path: '/telemedicine',
      roles: ['admin', 'doctor', 'patient'],
      badge: 'Live',
    },
    {
      text: 'Reports & Analytics',
      icon: <AssessmentIcon />,
      path: '/reports',
      roles: ['admin', 'doctor'],
      badge: 'Data',
    },
  ];

  const adminItems = [
    {
      text: 'Admin Panel',
      icon: <AdminIcon />,
      path: '/admin',
      roles: ['admin'],
    },
    {
      text: 'System Analytics',
      icon: <AnalyticsIcon />,
      path: '/analytics',
      roles: ['admin'],
    },
  ];

  const handleNavigation = (path) => {
    navigate(path);
  };

  const isActive = (path) => {
    return location.pathname === path;
  };

  const filteredMenuItems = menuItems.filter(item =>
    item.roles.includes(userRole)
  );

  const filteredAdminItems = adminItems.filter(item =>
    item.roles.includes(userRole)
  );

  return (
    <Drawer
      variant="persistent"
      anchor="left"
      open={open}
      sx={{
        width: drawerWidth,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: drawerWidth,
          boxSizing: 'border-box',
          borderRight: '1px solid rgba(0, 0, 0, 0.12)',
          backgroundColor: '#fafafa',
        },
      }}
    >
      <Box sx={{ p: 2, borderBottom: '1px solid rgba(0, 0, 0, 0.12)' }}>
        <Typography variant="h6" color="primary" fontWeight="bold">
          AI Medicine Platform
        </Typography>
        <Typography variant="caption" color="text.secondary">
          Personalized Healthcare Solutions
        </Typography>
      </Box>

      <List sx={{ pt: 1 }}>
        {filteredMenuItems.map((item) => (
          <ListItem key={item.text} disablePadding>
            <ListItemButton
              onClick={() => handleNavigation(item.path)}
              selected={isActive(item.path)}
              sx={{
                mx: 1,
                mb: 0.5,
                borderRadius: 1,
                '&.Mui-selected': {
                  backgroundColor: 'primary.main',
                  color: 'primary.contrastText',
                  '& .MuiListItemIcon-root': {
                    color: 'primary.contrastText',
                  },
                  '&:hover': {
                    backgroundColor: 'primary.dark',
                  },
                },
              }}
            >
              <ListItemIcon sx={{ minWidth: 40 }}>
                {item.icon}
              </ListItemIcon>
              <ListItemText
                primary={item.text}
                primaryTypographyProps={{
                  fontSize: '0.9rem',
                  fontWeight: isActive(item.path) ? 600 : 400,
                }}
              />
              {item.badge && (
                <Chip
                  label={item.badge}
                  size="small"
                  color={isActive(item.path) ? 'default' : 'primary'}
                  variant="outlined"
                  sx={{ fontSize: '0.7rem', height: 18 }}
                />
              )}
            </ListItemButton>
          </ListItem>
        ))}

        {filteredAdminItems.length > 0 && (
          <>
            <Divider sx={{ my: 2 }} />
            <Typography
              variant="caption"
              sx={{
                px: 3,
                py: 1,
                color: 'text.secondary',
                fontWeight: 600,
                textTransform: 'uppercase',
                letterSpacing: 1,
              }}
            >
              Administration
            </Typography>
            {filteredAdminItems.map((item) => (
              <ListItem key={item.text} disablePadding>
                <ListItemButton
                  onClick={() => handleNavigation(item.path)}
                  selected={isActive(item.path)}
                  sx={{
                    mx: 1,
                    mb: 0.5,
                    borderRadius: 1,
                    '&.Mui-selected': {
                      backgroundColor: 'secondary.main',
                      color: 'secondary.contrastText',
                      '& .MuiListItemIcon-root': {
                        color: 'secondary.contrastText',
                      },
                      '&:hover': {
                        backgroundColor: 'secondary.dark',
                      },
                    },
                  }}
                >
                  <ListItemIcon sx={{ minWidth: 40 }}>
                    {item.icon}
                  </ListItemIcon>
                  <ListItemText
                    primary={item.text}
                    primaryTypographyProps={{
                      fontSize: '0.9rem',
                      fontWeight: isActive(item.path) ? 600 : 400,
                    }}
                  />
                </ListItemButton>
              </ListItem>
            ))}
          </>
        )}
      </List>

      <Box sx={{ mt: 'auto', p: 2, borderTop: '1px solid rgba(0, 0, 0, 0.12)' }}>
        <Typography variant="caption" color="text.secondary" display="block">
          Version 1.0.0
        </Typography>
        <Typography variant="caption" color="text.secondary" display="block">
          © 2024 AI Medicine Platform
        </Typography>
      </Box>
    </Drawer>
  );
};

export default Sidebar;
