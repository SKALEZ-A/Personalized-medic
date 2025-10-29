/**
 * PWA Install Prompt Component for AI Personalized Medicine Platform
 * Handles Progressive Web App installation and offline functionality
 */

import React, { useState, useEffect } from 'react';
import './PWAInstallPrompt.css';

const PWAInstallPrompt = () => {
  const [deferredPrompt, setDeferredPrompt] = useState(null);
  const [showInstallPrompt, setShowInstallPrompt] = useState(false);
  const [isInstalled, setIsInstalled] = useState(false);
  const [isOnline, setIsOnline] = useState(navigator.onLine);
  const [showOfflineBanner, setShowOfflineBanner] = useState(false);
  const [cachedData, setCachedData] = useState(null);

  // PWA Install Prompt Logic
  useEffect(() => {
    // Check if already installed
    if (window.matchMedia('(display-mode: standalone)').matches ||
        window.navigator.standalone === true) {
      setIsInstalled(true);
      return;
    }

    // Listen for the beforeinstallprompt event
    const handleBeforeInstallPrompt = (e) => {
      // Prevent the mini-infobar from appearing on mobile
      e.preventDefault();
      // Stash the event so it can be triggered later
      setDeferredPrompt(e);

      // Show install prompt after a delay (don't be too aggressive)
      setTimeout(() => {
        if (!localStorage.getItem('pwa-install-dismissed')) {
          setShowInstallPrompt(true);
        }
      }, 30000); // Show after 30 seconds
    };

    // Listen for successful installation
    const handleAppInstalled = () => {
      setIsInstalled(true);
      setShowInstallPrompt(false);
      setDeferredPrompt(null);

      // Show success message
      if ('serviceWorker' in navigator) {
        navigator.serviceWorker.controller?.postMessage({
          type: 'SHOW_NOTIFICATION',
          title: 'MediCare AI Installed!',
          body: 'The app is now installed and ready to use offline.',
          icon: '/icons/icon-192x192.png'
        });
      }
    };

    window.addEventListener('beforeinstallprompt', handleBeforeInstallPrompt);
    window.addEventListener('appinstalled', handleAppInstalled);

    return () => {
      window.removeEventListener('beforeinstallprompt', handleBeforeInstallPrompt);
      window.removeEventListener('appinstalled', handleAppInstalled);
    };
  }, []);

  // Online/Offline Detection
  useEffect(() => {
    const handleOnline = () => {
      setIsOnline(true);
      setShowOfflineBanner(false);

      // Sync cached data when back online
      if (cachedData) {
        syncCachedData();
      }
    };

    const handleOffline = () => {
      setIsOnline(false);
      setShowOfflineBanner(true);

      // Load cached data for offline use
      loadCachedData();
    };

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, [cachedData]);

  // Service Worker Registration
  useEffect(() => {
    if ('serviceWorker' in navigator) {
      navigator.serviceWorker
        .register('/service-worker.js')
        .then((registration) => {
          console.log('SW registered: ', registration);

          // Listen for messages from service worker
          navigator.serviceWorker.addEventListener('message', (event) => {
            const { type, data } = event.data;

            switch (type) {
              case 'CACHE_UPDATED':
                console.log('Cache updated:', data);
                break;
              case 'OFFLINE_READY':
                console.log('App ready for offline use');
                break;
              case 'SYNC_SUCCESS':
                handleSyncSuccess(data);
                break;
              default:
                console.log('Unknown SW message:', type);
            }
          });
        })
        .catch((registrationError) => {
          console.log('SW registration failed: ', registrationError);
        });
    }
  }, []);

  const handleInstallClick = async () => {
    if (!deferredPrompt) return;

    // Show the install prompt
    deferredPrompt.prompt();

    // Wait for the user to respond to the prompt
    const { outcome } = await deferredPrompt.userChoice;

    // Reset the deferred prompt variable
    setDeferredPrompt(null);

    if (outcome === 'accepted') {
      console.log('User accepted the install prompt');
    } else {
      console.log('User dismissed the install prompt');
    }

    setShowInstallPrompt(false);
  };

  const dismissInstallPrompt = () => {
    setShowInstallPrompt(false);
    localStorage.setItem('pwa-install-dismissed', 'true');
  };

  const loadCachedData = async () => {
    if ('serviceWorker' in navigator && navigator.serviceWorker.controller) {
      navigator.serviceWorker.controller.postMessage({
        type: 'GET_CACHED_DATA'
      });

      // Listen for response
      const handleMessage = (event) => {
        if (event.data.type === 'CACHED_DATA_RESPONSE') {
          setCachedData(event.data.data);
          navigator.serviceWorker.removeEventListener('message', handleMessage);
        }
      };

      navigator.serviceWorker.addEventListener('message', handleMessage);
    }
  };

  const syncCachedData = async () => {
    if ('serviceWorker' in navigator && navigator.serviceWorker.controller) {
      navigator.serviceWorker.controller.postMessage({
        type: 'SYNC_CACHED_DATA'
      });
    }
  };

  const handleSyncSuccess = (data) => {
    console.log('Data sync successful:', data);
    // Update UI to reflect synced data
    setCachedData(null);
  };

  const requestNotificationPermission = async () => {
    if ('Notification' in window && 'serviceWorker' in navigator) {
      const permission = await Notification.requestPermission();

      if (permission === 'granted') {
        console.log('Notification permission granted');

        // Register for push notifications
        const registration = await navigator.serviceWorker.ready;
        const subscription = await registration.pushManager.subscribe({
          userVisibleOnly: true,
          applicationServerKey: urlBase64ToUint8Array(
            'BNcRdreALRFXTkOOUHK1EtK2wtaz5Ry4YfYCA_0QTpQtUbVlUls0VJXg7A8wR' +
            'I5c4Jj5VzH5j0q_J0qJc5v8n0w' // Replace with your actual VAPID key
          )
        });

        // Send subscription to server
        console.log('Push subscription:', subscription);
      }
    }
  };

  // Utility function to convert VAPID key
  const urlBase64ToUint8Array = (base64String) => {
    const padding = '='.repeat((4 - base64String.length % 4) % 4);
    const base64 = (base64String + padding)
      .replace(/-/g, '+')
      .replace(/_/g, '/');

    const rawData = window.atob(base64);
    const outputArray = new Uint8Array(rawData.length);

    for (let i = 0; i < rawData.length; ++i) {
      outputArray[i] = rawData.charCodeAt(i);
    }
    return outputArray;
  };

  return (
    <>
      {/* Offline Banner */}
      {showOfflineBanner && (
        <div className="pwa-offline-banner" role="alert" aria-live="assertive">
          <div className="offline-content">
            <span className="offline-icon">📶</span>
            <div className="offline-text">
              <strong>You're offline</strong>
              <span>You can still access some features using cached data.</span>
            </div>
            <button
              className="offline-close"
              onClick={() => setShowOfflineBanner(false)}
              aria-label="Dismiss offline notification"
            >
              ✕
            </button>
          </div>
        </div>
      )}

      {/* PWA Install Prompt */}
      {showInstallPrompt && !isInstalled && (
        <div className="pwa-install-prompt" role="dialog" aria-labelledby="install-title">
          <div className="install-content">
            <div className="install-icon">🏥</div>
            <div className="install-text">
              <h3 id="install-title">Install MediCare AI</h3>
              <p>
                Install our app for a better experience with offline access,
                push notifications, and quick access from your home screen.
              </p>
            </div>
            <div className="install-actions">
              <button
                className="install-btn"
                onClick={handleInstallClick}
                aria-label="Install MediCare AI app"
              >
                Install
              </button>
              <button
                className="dismiss-btn"
                onClick={dismissInstallPrompt}
                aria-label="Dismiss install prompt"
              >
                Not Now
              </button>
            </div>
          </div>
        </div>
      )}

      {/* PWA Status Indicator */}
      <div className="pwa-status">
        <button
          className="status-btn"
          onClick={() => setShowInstallPrompt(true)}
          disabled={isInstalled}
          aria-label={
            isInstalled
              ? "App is installed"
              : isOnline
                ? "Install app for offline access"
                : "Offline - cached data available"
          }
        >
          {isInstalled ? (
            <>✅ Installed</>
          ) : isOnline ? (
            <>📱 Install App</>
          ) : (
            <>📶 Offline Mode</>
          )}
        </button>

        {/* Notification Permission Request */}
        {!isInstalled && 'Notification' in window && Notification.permission === 'default' && (
          <button
            className="notification-btn"
            onClick={requestNotificationPermission}
            aria-label="Enable push notifications"
          >
            🔔 Enable Notifications
          </button>
        )}
      </div>

      {/* Offline Data Indicator */}
      {cachedData && (
        <div className="cached-data-indicator" role="status" aria-live="polite">
          <span className="cached-icon">💾</span>
          <span>Using cached data from {new Date(cachedData.timestamp).toLocaleString()}</span>
          <button
            className="sync-btn"
            onClick={syncCachedData}
            disabled={!isOnline}
            aria-label="Sync cached data"
          >
            🔄 Sync
          </button>
        </div>
      )}

      {/* PWA Update Available Banner */}
      <PWAUpdatePrompt />
    </>
  );
};

// PWA Update Prompt Component
const PWAUpdatePrompt = () => {
  const [showUpdatePrompt, setShowUpdatePrompt] = useState(false);
  const [newWorker, setNewWorker] = useState(null);

  useEffect(() => {
    if ('serviceWorker' in navigator) {
      navigator.serviceWorker.addEventListener('message', (event) => {
        if (event.data.type === 'UPDATE_AVAILABLE') {
          setNewWorker(event.data.worker);
          setShowUpdatePrompt(true);
        }
      });
    }
  }, []);

  const handleUpdate = () => {
    if (newWorker) {
      newWorker.postMessage({ type: 'SKIP_WAITING' });
      window.location.reload();
    }
  };

  const dismissUpdate = () => {
    setShowUpdatePrompt(false);
  };

  if (!showUpdatePrompt) return null;

  return (
    <div className="pwa-update-prompt" role="dialog" aria-labelledby="update-title">
      <div className="update-content">
        <div className="update-icon">🔄</div>
        <div className="update-text">
          <h3 id="update-title">Update Available</h3>
          <p>A new version of MediCare AI is available. Update now for the latest features and improvements.</p>
        </div>
        <div className="update-actions">
          <button
            className="update-btn"
            onClick={handleUpdate}
            aria-label="Update to latest version"
          >
            Update Now
          </button>
          <button
            className="dismiss-btn"
            onClick={dismissUpdate}
            aria-label="Dismiss update prompt"
          >
            Later
          </button>
        </div>
      </div>
    </div>
  );
};

// PWA Feature Detector Hook
export const usePWAFeatures = () => {
  const [features, setFeatures] = useState({
    isInstallable: false,
    isInstalled: false,
    canNotify: false,
    isOnline: navigator.onLine,
    hasServiceWorker: false,
    hasBackgroundSync: false,
    hasPushManager: false
  });

  useEffect(() => {
    // Check PWA capabilities
    const checkFeatures = () => {
      setFeatures({
        isInstallable: 'beforeinstallprompt' in window,
        isInstalled: window.matchMedia('(display-mode: standalone)').matches ||
                    window.navigator.standalone === true,
        canNotify: 'Notification' in window,
        isOnline: navigator.onLine,
        hasServiceWorker: 'serviceWorker' in navigator,
        hasBackgroundSync: 'serviceWorker' in navigator && 'sync' in window.ServiceWorkerRegistration.prototype,
        hasPushManager: 'serviceWorker' in navigator && 'PushManager' in window
      });
    };

    checkFeatures();

    // Listen for online/offline changes
    const handleOnline = () => setFeatures(prev => ({ ...prev, isOnline: true }));
    const handleOffline = () => setFeatures(prev => ({ ...prev, isOnline: false }));

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  return features;
};

// Offline Data Manager Hook
export const useOfflineData = () => {
  const [offlineData, setOfflineData] = useState({
    healthRecords: [],
    appointments: [],
    medications: [],
    lastSync: null
  });

  const saveOfflineData = (type, data) => {
    const updatedData = { ...offlineData };
    updatedData[type] = data;
    updatedData.lastSync = new Date().toISOString();
    setOfflineData(updatedData);

    // Cache in service worker
    if ('serviceWorker' in navigator && navigator.serviceWorker.controller) {
      navigator.serviceWorker.controller.postMessage({
        type: 'CACHE_OFFLINE_DATA',
        data: updatedData
      });
    }

    // Also save to localStorage as backup
    localStorage.setItem('offline-health-data', JSON.stringify(updatedData));
  };

  const loadOfflineData = () => {
    const cached = localStorage.getItem('offline-health-data');
    if (cached) {
      try {
        setOfflineData(JSON.parse(cached));
      } catch (error) {
        console.error('Failed to load offline data:', error);
      }
    }
  };

  const syncOfflineData = async () => {
    if ('serviceWorker' in navigator && navigator.serviceWorker.controller) {
      navigator.serviceWorker.controller.postMessage({
        type: 'SYNC_OFFLINE_DATA'
      });
    }
  };

  useEffect(() => {
    loadOfflineData();
  }, []);

  return {
    offlineData,
    saveOfflineData,
    loadOfflineData,
    syncOfflineData
  };
};

export default PWAInstallPrompt;
