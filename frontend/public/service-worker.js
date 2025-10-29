/**
 * Service Worker for AI Personalized Medicine Platform PWA
 * Handles caching, offline functionality, background sync, and push notifications
 */

const CACHE_NAME = 'medicare-ai-v1.0.0';
const STATIC_CACHE = 'medicare-ai-static-v1.0.0';
const DYNAMIC_CACHE = 'medicare-ai-dynamic-v1.0.0';
const API_CACHE = 'medicare-ai-api-v1.0.0';

// Resources to cache immediately
const STATIC_ASSETS = [
  '/',
  '/static/js/bundle.js',
  '/static/css/main.css',
  '/manifest.json',
  '/favicon.ico',
  '/offline.html'
];

// API endpoints to cache
const API_ENDPOINTS = [
  '/api/health',
  '/api/user/profile',
  '/api/health/metrics',
  '/api/appointments/upcoming'
];

// Install event - cache static assets
self.addEventListener('install', (event) => {
  console.log('[Service Worker] Installing...');

  event.waitUntil(
    caches.open(STATIC_CACHE)
      .then((cache) => {
        console.log('[Service Worker] Caching static assets');
        return cache.addAll(STATIC_ASSETS);
      })
      .catch((error) => {
        console.error('[Service Worker] Cache installation failed:', error);
      })
  );

  // Force activation of new service worker
  self.skipWaiting();
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
  console.log('[Service Worker] Activating...');

  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => {
          if (cacheName !== STATIC_CACHE &&
              cacheName !== DYNAMIC_CACHE &&
              cacheName !== API_CACHE &&
              !cacheName.startsWith('medicare-ai-')) {
            console.log('[Service Worker] Deleting old cache:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    })
  );

  // Take control of all clients immediately
  self.clients.claim();
});

// Fetch event - handle requests
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // Handle different types of requests
  if (url.origin === location.origin) {
    // Same origin requests
    if (request.destination === 'document') {
      // HTML pages - Network first, fallback to cache
      event.respondWith(
        fetch(request)
          .catch(() => {
            return caches.match('/offline.html') || caches.match('/');
          })
      );
    } else if (STATIC_ASSETS.some(asset => url.pathname.endsWith(asset))) {
      // Static assets - Cache first
      event.respondWith(
        caches.match(request)
          .then((response) => {
            return response || fetch(request).then((response) => {
              // Cache the new response
              const responseClone = response.clone();
              caches.open(STATIC_CACHE).then((cache) => {
                cache.put(request, responseClone);
              });
              return response;
            });
          })
      );
    } else {
      // Other same-origin requests - Network first, cache fallback
      event.respondWith(
        fetch(request)
          .then((response) => {
            // Cache successful responses
            if (response.status === 200) {
              const responseClone = response.clone();
              caches.open(DYNAMIC_CACHE).then((cache) => {
                cache.put(request, responseClone);
              });
            }
            return response;
          })
          .catch(() => {
            return caches.match(request);
          })
      );
    }
  } else if (API_ENDPOINTS.some(endpoint => url.pathname.startsWith(endpoint))) {
    // API requests - Network first with background sync fallback
    event.respondWith(
      fetch(request)
        .then((response) => {
          // Cache successful API responses
          if (response.status === 200) {
            const responseClone = response.clone();
            caches.open(API_CACHE).then((cache) => {
              // Add timestamp for cache invalidation
              const cacheRequest = new Request(request.url, {
                headers: { ...request.headers, 'sw-cache-time': Date.now() }
              });
              cache.put(cacheRequest, responseClone);
            });
          }
          return response;
        })
        .catch(() => {
          // Return cached API response if available
          return caches.match(request).then((cachedResponse) => {
            if (cachedResponse) {
              return cachedResponse;
            }
            // Return offline API response
            return new Response(
              JSON.stringify({
                error: 'Offline',
                message: 'You are currently offline. This data may be outdated.',
                cached: true,
                timestamp: new Date().toISOString()
              }),
              {
                status: 200,
                headers: { 'Content-Type': 'application/json' }
              }
            );
          });
        })
    );
  } else {
    // External requests - Cache first for performance
    event.respondWith(
      caches.match(request)
        .then((response) => {
          return response || fetch(request).then((response) => {
            // Cache external resources (images, fonts, etc.)
            if (response.status === 200 &&
                (request.destination === 'image' ||
                 request.destination === 'font' ||
                 request.destination === 'style' ||
                 request.destination === 'script')) {
              const responseClone = response.clone();
              caches.open(DYNAMIC_CACHE).then((cache) => {
                cache.put(request, responseClone);
              });
            }
            return response;
          });
        })
    );
  }
});

// Background sync for offline actions
self.addEventListener('sync', (event) => {
  console.log('[Service Worker] Background sync:', event.tag);

  if (event.tag === 'health-data-sync') {
    event.waitUntil(syncHealthData());
  } else if (event.tag === 'appointment-sync') {
    event.waitUntil(syncAppointments());
  } else if (event.tag === 'medication-sync') {
    event.waitUntil(syncMedications());
  }
});

// Push notifications
self.addEventListener('push', (event) => {
  console.log('[Service Worker] Push received');

  let data = {};
  if (event.data) {
    data = event.data.json();
  }

  const options = {
    body: data.body || 'You have a new health update',
    icon: '/icons/icon-192x192.png',
    badge: '/icons/badge-72x72.png',
    vibrate: [100, 50, 100],
    data: {
      dateOfArrival: Date.now(),
      primaryKey: data.primaryKey || 1,
      url: data.url || '/'
    },
    actions: [
      {
        action: 'view',
        title: 'View Details',
        icon: '/icons/view-action.png'
      },
      {
        action: 'dismiss',
        title: 'Dismiss'
      }
    ],
    requireInteraction: data.urgent || false,
    silent: data.silent || false,
    tag: data.tag || 'health-notification'
  };

  event.waitUntil(
    self.registration.showNotification(
      data.title || 'MediCare AI',
      options
    )
  );
});

// Notification click handler
self.addEventListener('notificationclick', (event) => {
  console.log('[Service Worker] Notification clicked');

  event.notification.close();

  if (event.action === 'view') {
    // Open the app and navigate to the relevant page
    event.waitUntil(
      clients.openWindow(event.notification.data.url || '/')
    );
  } else if (event.action === 'dismiss') {
    // Just dismiss - no action needed
    return;
  } else {
    // Default action - open the app
    event.waitUntil(
      clients.openWindow('/')
    );
  }
});

// Message handler for communication with the main thread
self.addEventListener('message', (event) => {
  const { type, data } = event.data;

  switch (type) {
    case 'SKIP_WAITING':
      self.skipWaiting();
      break;

    case 'CACHE_HEALTH_DATA':
      cacheHealthData(data);
      break;

    case 'GET_CACHED_DATA':
      getCachedData().then((cachedData) => {
        event.ports[0].postMessage(cachedData);
      });
      break;

    case 'CLEAR_CACHE':
      clearCache();
      break;

    default:
      console.log('[Service Worker] Unknown message type:', type);
  }
});

// Periodic background tasks
self.addEventListener('periodicsync', (event) => {
  if (event.tag === 'health-check') {
    event.waitUntil(performHealthCheck());
  }
});

// Background sync functions
async function syncHealthData() {
  try {
    console.log('[Service Worker] Syncing health data...');

    // Get cached offline actions
    const cache = await caches.open('offline-actions');
    const requests = await cache.keys();

    for (const request of requests) {
      try {
        await fetch(request);
        await cache.delete(request);
      } catch (error) {
        console.error('[Service Worker] Failed to sync request:', error);
      }
    }

    // Notify user of successful sync
    await self.registration.showNotification('MediCare AI', {
      body: 'Health data synchronized successfully',
      icon: '/icons/icon-192x192.png',
      tag: 'sync-success'
    });

  } catch (error) {
    console.error('[Service Worker] Health data sync failed:', error);
  }
}

async function syncAppointments() {
  console.log('[Service Worker] Syncing appointments...');
  // Implementation for appointment synchronization
}

async function syncMedications() {
  console.log('[Service Worker] Syncing medications...');
  // Implementation for medication synchronization
}

async function performHealthCheck() {
  try {
    console.log('[Service Worker] Performing periodic health check...');

    // Check API health
    const healthResponse = await fetch('/api/health');
    const healthData = await healthResponse.json();

    if (healthData.status !== 'healthy') {
      await self.registration.showNotification('MediCare AI', {
        body: 'System health check detected issues',
        icon: '/icons/icon-192x192.png',
        tag: 'health-alert',
        requireInteraction: true
      });
    }

  } catch (error) {
    console.error('[Service Worker] Health check failed:', error);
  }
}

// Cache management functions
async function cacheHealthData(data) {
  const cache = await caches.open('health-data-cache');
  const request = new Request('/api/health/cached', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data)
  });

  await cache.put(request, new Response(JSON.stringify(data)));
}

async function getCachedData() {
  const cache = await caches.open('health-data-cache');
  const request = new Request('/api/health/cached');
  const response = await cache.match(request);

  if (response) {
    return await response.json();
  }

  return null;
}

async function clearCache() {
  const cacheNames = await caches.keys();
  await Promise.all(
    cacheNames.map(cacheName => caches.delete(cacheName))
  );
}

// Cache size management
async function manageCacheSize() {
  const cache = await caches.open(DYNAMIC_CACHE);
  const keys = await cache.keys();

  if (keys.length > 100) {
    // Remove oldest entries
    const entriesToDelete = keys.slice(0, keys.length - 100);
    await Promise.all(
      entriesToDelete.map(request => cache.delete(request))
    );
  }
}

// Performance monitoring
const performanceMarks = new Map();

function markPerformance(name) {
  performanceMarks.set(name, performance.now());
}

function measurePerformance(name) {
  if (performanceMarks.has(name)) {
    const startTime = performanceMarks.get(name);
    const duration = performance.now() - startTime;
    console.log(`[Performance] ${name}: ${duration.toFixed(2)}ms`);
    performanceMarks.delete(name);
  }
}

// Error tracking
function trackError(error, context = {}) {
  const errorData = {
    message: error.message,
    stack: error.stack,
    timestamp: new Date().toISOString(),
    context: context,
    userAgent: navigator.userAgent,
    url: self.location.href
  };

  // Store error for later analysis
  // In a real implementation, this would send to an error tracking service
  console.error('[Service Worker] Error tracked:', errorData);
}

// Install periodic cache management
setInterval(manageCacheSize, 5 * 60 * 1000); // Every 5 minutes

console.log('[Service Worker] Initialized successfully');
