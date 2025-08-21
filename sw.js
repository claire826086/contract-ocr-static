self.addEventListener("install", e => {
  self.skipWaiting();
  e.waitUntil(caches.open("v1").then(cache => cache.addAll(["./", "./index.html", "./app.js", "./style.css"])));
});
self.addEventListener("fetch", e => {
  e.respondWith(caches.match(e.request).then(r => r || fetch(e.request)));
});