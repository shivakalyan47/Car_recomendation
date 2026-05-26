const CACHE_NAME = 'car-recs-v1';
const ASSETS = [
  './',
  './index.html',
  './index.css',
  './index.js',
  './manifest.json',
  './assets/app_icon.png',
  './data/cars.csv'
];

// Install Event
self.addEventListener('install', (e) => {
  e.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log('[Service Worker] Caching all static assets');
      return cache.addAll(ASSETS);
    }).then(() => self.skipWaiting())
  );
});

// Activate Event
self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys().then((keys) => {
      return Promise.all(
        keys.map((key) => {
          if (key !== CACHE_NAME) {
            console.log('[Service Worker] Removing old cache', key);
            return caches.delete(key);
          }
        })
      );
    }).then(() => self.clients.claim())
  );
});

// Fetch Event (Cache-first with network fallback)
self.addEventListener('fetch', (e) => {
  // Only handle GET requests and local/CDN assets
  if (e.request.method !== 'GET') return;
  
  e.respondWith(
    caches.match(e.request).then((cachedResponse) => {
      if (cachedResponse) {
        return cachedResponse;
      }
      
      return fetch(e.request).then((networkResponse) => {
        if (!networkResponse || networkResponse.status !== 200) {
          return networkResponse;
        }
        
        // Cache new fetch requests dynamically (like Plotly CDN)
        const responseToCache = networkResponse.clone();
        caches.open(CACHE_NAME).then((cache) => {
          // Do not cache external APIs if any, but cache static CDNs
          if (e.request.url.startsWith('http') || e.request.url.startsWith('https')) {
            cache.put(e.request, responseToCache);
          }
        });
        
        return networkResponse;
      });
    }).catch(() => {
      // Offline fallback if network fails
      console.log('[Service Worker] Resource fetch failed offline');
    })
  );
});
