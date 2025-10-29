import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import axios from 'axios';

// Types
interface User {
  id: string;
  username: string;
  email: string;
  firstName: string;
  lastName: string;
  role: 'patient' | 'physician' | 'nurse' | 'pharmacist' | 'researcher' | 'admin';
  permissions: string[];
  mfaEnabled: boolean;
  profileComplete: boolean;
  lastLogin: string;
}

interface AuthState {
  user: User | null;
  token: string | null;
  refreshToken: string | null;
  isAuthenticated: boolean;
  loading: boolean;
  mfaRequired: boolean;
  mfaVerified: boolean;
}

interface AuthContextType extends AuthState {
  login: (username: string, password: string, rememberMe?: boolean) => Promise<void>;
  logout: () => Promise<void>;
  register: (userData: RegisterData) => Promise<void>;
  refreshToken: () => Promise<void>;
  verifyMFA: (code: string) => Promise<void>;
  setupMFA: () => Promise<MFAData>;
  disableMFA: () => Promise<void>;
  forgotPassword: (email: string) => Promise<void>;
  resetPassword: (token: string, newPassword: string) => Promise<void>;
  updateProfile: (profileData: Partial<User>) => Promise<void>;
  changePassword: (currentPassword: string, newPassword: string) => Promise<void>;
}

interface RegisterData {
  username: string;
  email: string;
  password: string;
  firstName: string;
  lastName: string;
  role: User['role'];
  dateOfBirth: string;
  phone?: string;
  address?: Address;
}

interface Address {
  street: string;
  city: string;
  state: string;
  zipCode: string;
  country: string;
}

interface MFAData {
  secret: string;
  qrCodeUrl: string;
  backupCodes: string[];
}

// Context
const AuthContext = createContext<AuthContextType | undefined>(undefined);

// API Configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

// Request interceptor to add auth token
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('authToken');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor to handle token refresh
api.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config;

    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true;

      try {
        const refreshToken = localStorage.getItem('refreshToken');
        if (refreshToken) {
          const response = await axios.post(`${API_BASE_URL}/auth/refresh`, {
            refresh_token: refreshToken,
          });

          const { access_token } = response.data;
          localStorage.setItem('authToken', access_token);

          originalRequest.headers.Authorization = `Bearer ${access_token}`;
          return api(originalRequest);
        }
      } catch (refreshError) {
        // Refresh failed, logout user
        localStorage.removeItem('authToken');
        localStorage.removeItem('refreshToken');
        window.location.href = '/login';
      }
    }

    return Promise.reject(error);
  }
);

// Provider Component
interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [authState, setAuthState] = useState<AuthState>({
    user: null,
    token: null,
    refreshToken: null,
    isAuthenticated: false,
    loading: true,
    mfaRequired: false,
    mfaVerified: false,
  });

  // Initialize auth state on mount
  useEffect(() => {
    const initializeAuth = async () => {
      const token = localStorage.getItem('authToken');
      const refreshToken = localStorage.getItem('refreshToken');
      const userData = localStorage.getItem('userData');

      if (token && userData) {
        try {
          // Verify token is still valid
          const response = await api.get('/auth/verify');
          const user = JSON.parse(userData);

          setAuthState({
            user,
            token,
            refreshToken,
            isAuthenticated: true,
            loading: false,
            mfaRequired: user.mfaEnabled && !localStorage.getItem('mfaVerified'),
            mfaVerified: !!localStorage.getItem('mfaVerified'),
          });
        } catch (error) {
          // Token invalid, clear storage
          localStorage.removeItem('authToken');
          localStorage.removeItem('refreshToken');
          localStorage.removeItem('userData');
          localStorage.removeItem('mfaVerified');

          setAuthState(prev => ({ ...prev, loading: false }));
        }
      } else {
        setAuthState(prev => ({ ...prev, loading: false }));
      }
    };

    initializeAuth();
  }, []);

  // Login function
  const login = async (username: string, password: string, rememberMe = false) => {
    try {
      setAuthState(prev => ({ ...prev, loading: true }));

      const response = await api.post('/auth/login', {
        username,
        password,
        remember_me: rememberMe,
      });

      const { access_token, refresh_token, user, mfa_required } = response.data;

      if (mfa_required) {
        setAuthState(prev => ({
          ...prev,
          user,
          token: access_token,
          refreshToken: refresh_token,
          mfaRequired: true,
          mfaVerified: false,
          loading: false,
        }));

        // Store temporarily until MFA is verified
        sessionStorage.setItem('tempAuthToken', access_token);
        sessionStorage.setItem('tempRefreshToken', refresh_token);
        sessionStorage.setItem('tempUserData', JSON.stringify(user));
      } else {
        // Store auth data
        localStorage.setItem('authToken', access_token);
        localStorage.setItem('refreshToken', refresh_token);
        localStorage.setItem('userData', JSON.stringify(user));

        setAuthState({
          user,
          token: access_token,
          refreshToken: refresh_token,
          isAuthenticated: true,
          loading: false,
          mfaRequired: false,
          mfaVerified: true,
        });
      }
    } catch (error: any) {
      setAuthState(prev => ({ ...prev, loading: false }));

      if (error.response?.status === 429) {
        throw new Error('Too many failed login attempts. Please try again later.');
      } else if (error.response?.status === 401) {
        throw new Error('Invalid username or password.');
      } else {
        throw new Error('Login failed. Please try again.');
      }
    }
  };

  // Verify MFA function
  const verifyMFA = async (code: string) => {
    try {
      const tempToken = sessionStorage.getItem('tempAuthToken');
      if (!tempToken) {
        throw new Error('No pending MFA verification');
      }

      await api.post('/auth/mfa/verify', { code }, {
        headers: { Authorization: `Bearer ${tempToken}` }
      });

      // MFA verified, complete login
      const tempRefreshToken = sessionStorage.getItem('tempRefreshToken');
      const tempUserData = sessionStorage.getItem('tempUserData');

      if (tempRefreshToken && tempUserData) {
        const user = JSON.parse(tempUserData);

        localStorage.setItem('authToken', tempToken);
        localStorage.setItem('refreshToken', tempRefreshToken);
        localStorage.setItem('userData', tempUserData);
        localStorage.setItem('mfaVerified', 'true');

        // Clear temp data
        sessionStorage.removeItem('tempAuthToken');
        sessionStorage.removeItem('tempRefreshToken');
        sessionStorage.removeItem('tempUserData');

        setAuthState({
          user,
          token: tempToken,
          refreshToken: tempRefreshToken,
          isAuthenticated: true,
          loading: false,
          mfaRequired: false,
          mfaVerified: true,
        });
      }
    } catch (error: any) {
      if (error.response?.status === 401) {
        throw new Error('Invalid MFA code');
      } else {
        throw new Error('MFA verification failed');
      }
    }
  };

  // Setup MFA function
  const setupMFA = async (): Promise<MFAData> => {
    try {
      const response = await api.post('/auth/mfa/setup');
      return response.data;
    } catch (error: any) {
      throw new Error('Failed to setup MFA');
    }
  };

  // Disable MFA function
  const disableMFA = async () => {
    try {
      await api.delete('/auth/mfa/disable');

      if (authState.user) {
        const updatedUser = { ...authState.user, mfaEnabled: false };
        localStorage.setItem('userData', JSON.stringify(updatedUser));

        setAuthState(prev => ({
          ...prev,
          user: updatedUser,
          mfaRequired: false,
          mfaVerified: false,
        }));

        localStorage.removeItem('mfaVerified');
      }
    } catch (error: any) {
      throw new Error('Failed to disable MFA');
    }
  };

  // Logout function
  const logout = async () => {
    try {
      await api.post('/auth/logout');
    } catch (error) {
      // Ignore logout API errors
    } finally {
      // Clear all auth data
      localStorage.removeItem('authToken');
      localStorage.removeItem('refreshToken');
      localStorage.removeItem('userData');
      localStorage.removeItem('mfaVerified');
      sessionStorage.clear();

      setAuthState({
        user: null,
        token: null,
        refreshToken: null,
        isAuthenticated: false,
        loading: false,
        mfaRequired: false,
        mfaVerified: false,
      });
    }
  };

  // Register function
  const register = async (userData: RegisterData) => {
    try {
      setAuthState(prev => ({ ...prev, loading: true }));

      const response = await api.post('/auth/register', userData);
      const { user, access_token, refresh_token } = response.data;

      // Store auth data
      localStorage.setItem('authToken', access_token);
      localStorage.setItem('refreshToken', refresh_token);
      localStorage.setItem('userData', JSON.stringify(user));

      setAuthState({
        user,
        token: access_token,
        refreshToken: refresh_token,
        isAuthenticated: true,
        loading: false,
        mfaRequired: false,
        mfaVerified: true,
      });
    } catch (error: any) {
      setAuthState(prev => ({ ...prev, loading: false }));

      if (error.response?.data?.message) {
        throw new Error(error.response.data.message);
      } else {
        throw new Error('Registration failed. Please try again.');
      }
    }
  };

  // Refresh token function
  const refreshToken = async () => {
    try {
      const currentRefreshToken = authState.refreshToken || localStorage.getItem('refreshToken');
      if (!currentRefreshToken) {
        throw new Error('No refresh token available');
      }

      const response = await axios.post(`${API_BASE_URL}/auth/refresh`, {
        refresh_token: currentRefreshToken,
      });

      const { access_token } = response.data;

      localStorage.setItem('authToken', access_token);

      setAuthState(prev => ({
        ...prev,
        token: access_token,
      }));
    } catch (error) {
      // Refresh failed, logout user
      await logout();
      throw error;
    }
  };

  // Forgot password function
  const forgotPassword = async (email: string) => {
    try {
      await api.post('/auth/forgot-password', { email });
    } catch (error: any) {
      // Don't reveal if email exists or not for security
      throw new Error('If the email exists, you will receive a password reset link.');
    }
  };

  // Reset password function
  const resetPassword = async (token: string, newPassword: string) => {
    try {
      await api.post('/auth/reset-password', {
        token,
        new_password: newPassword,
      });
    } catch (error: any) {
      if (error.response?.data?.message) {
        throw new Error(error.response.data.message);
      } else {
        throw new Error('Password reset failed');
      }
    }
  };

  // Update profile function
  const updateProfile = async (profileData: Partial<User>) => {
    try {
      const response = await api.put('/auth/profile', profileData);
      const updatedUser = response.data.user;

      localStorage.setItem('userData', JSON.stringify(updatedUser));

      setAuthState(prev => ({
        ...prev,
        user: updatedUser,
      }));
    } catch (error: any) {
      if (error.response?.data?.message) {
        throw new Error(error.response.data.message);
      } else {
        throw new Error('Profile update failed');
      }
    }
  };

  // Change password function
  const changePassword = async (currentPassword: string, newPassword: string) => {
    try {
      await api.post('/auth/change-password', {
        current_password: currentPassword,
        new_password: newPassword,
      });
    } catch (error: any) {
      if (error.response?.data?.message) {
        throw new Error(error.response.data.message);
      } else {
        throw new Error('Password change failed');
      }
    }
  };

  const contextValue: AuthContextType = {
    ...authState,
    login,
    logout,
    register,
    refreshToken,
    verifyMFA,
    setupMFA,
    disableMFA,
    forgotPassword,
    resetPassword,
    updateProfile,
    changePassword,
  };

  return (
    <AuthContext.Provider value={contextValue}>
      {children}
    </AuthContext.Provider>
  );
};

// Hook to use auth context
export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

// Export types for use in other components
export type { User, AuthState, RegisterData, Address, MFAData };
