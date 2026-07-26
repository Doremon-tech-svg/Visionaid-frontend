import { useEffect, useRef, useState, useCallback } from 'react';
import { Navigation, X, MapPin, Compass } from 'lucide-react';

// Loads Leaflet from CDN once — avoids an npm dependency + bundler config
// just for this one component. Safe to call repeatedly; no-ops after first load.
let leafletLoading = null;
function loadLeaflet() {
  if (window.L) return Promise.resolve(window.L);
  if (leafletLoading) return leafletLoading;
  leafletLoading = new Promise((resolve, reject) => {
    const css = document.createElement('link');
    css.rel = 'stylesheet';
    css.href = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css';
    document.head.appendChild(css);

    const script = document.createElement('script');
    script.src = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js';
    script.onload = () => resolve(window.L);
    script.onerror = () => reject(new Error('Leaflet failed to load from CDN'));
    document.head.appendChild(script);
  });
  return leafletLoading;
}

/**
 * Props:
 *  - onSetWaypoint(latlng)  called when user taps the map to drop a pin
 *  - onStartNav()           called when user confirms "Navigate here"
 *  - waypoint {lat,lng,name}?  current target, if any
 *  - routePath [{lat,lng}]?    full walking route to draw as a line
 *  - userPos {lat,lng}?        live GPS position
 *  - heading number|null       device compass heading in degrees, or null if unavailable
 *  - onClose()
 */
export default function WaypointMap({ onSetWaypoint, onStartNav, onStopNav, waypoint, routePath, userPos, heading, navigating, onClose }) {
  const mapDivRef = useRef(null);
  const mapRef = useRef(null);
  const userMarkerRef = useRef(null);
  const waypointMarkerRef = useRef(null);
  const routeLineRef = useRef(null);
  const [ready, setReady] = useState(false);
  const [pendingPin, setPendingPin] = useState(null); // {lat,lng} tapped but not yet named/confirmed
  const [pinName, setPinName] = useState('');

  useEffect(() => {
    let cancelled = false;
    loadLeaflet().then((L) => {
      if (cancelled || !mapDivRef.current || mapRef.current) return;
      const center = userPos ? [userPos.lat, userPos.lng] : [28.6139, 77.2090]; // fallback: Delhi, re-centers once GPS resolves
      const map = L.map(mapDivRef.current, { zoomControl: true }).setView(center, 17);
      L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors',
        maxZoom: 19,
      }).addTo(map);

      map.on('click', (e) => {
        setPendingPin({ lat: e.latlng.lat, lng: e.latlng.lng });
      });

      mapRef.current = map;
      setReady(true);
    }).catch((e) => console.error(e));

    return () => {
      cancelled = true;
      if (mapRef.current) { mapRef.current.remove(); mapRef.current = null; }
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // live user marker + auto-pan on first GPS fix
  useEffect(() => {
    if (!ready || !userPos || !window.L) return;
    const L = window.L;
    if (!userMarkerRef.current) {
      const icon = L.divIcon({
        className: '',
        html: `<div style="width:18px;height:18px;border-radius:50%;background:#3b82f6;border:3px solid white;box-shadow:0 0 0 4px rgba(59,130,246,0.3)"></div>`,
        iconSize: [18, 18],
      });
      userMarkerRef.current = L.marker([userPos.lat, userPos.lng], { icon, zIndexOffset: 1000 }).addTo(mapRef.current);
      mapRef.current.setView([userPos.lat, userPos.lng], 17);
    } else {
      userMarkerRef.current.setLatLng([userPos.lat, userPos.lng]);
    }
  }, [ready, userPos]);

  // waypoint marker
  useEffect(() => {
    if (!ready || !window.L) return;
    const L = window.L;
    if (waypointMarkerRef.current) { waypointMarkerRef.current.remove(); waypointMarkerRef.current = null; }
    if (waypoint) {
      const icon = L.divIcon({
        className: '',
        html: `<div style="font-size:28px;line-height:1;filter:drop-shadow(0 2px 2px rgba(0,0,0,0.4))">📍</div>`,
        iconSize: [28, 28],
        iconAnchor: [14, 28],
      });
      waypointMarkerRef.current = L.marker([waypoint.lat, waypoint.lng], { icon }).addTo(mapRef.current);
    }
  }, [ready, waypoint]);

  // route polyline
  useEffect(() => {
    if (!ready || !window.L) return;
    const L = window.L;
    if (routeLineRef.current) { routeLineRef.current.remove(); routeLineRef.current = null; }
    if (routePath && routePath.length > 1) {
      routeLineRef.current = L.polyline(routePath.map(p => [p.lat, p.lng]), { color: '#f59e0b', weight: 5, opacity: 0.85 }).addTo(mapRef.current);
      mapRef.current.fitBounds(routeLineRef.current.getBounds(), { padding: [40, 40] });
    }
  }, [ready, routePath]);

  const confirmPin = useCallback(() => {
    if (!pendingPin) return;
    onSetWaypoint({ ...pendingPin, name: pinName.trim() || 'Waypoint' });
    setPendingPin(null);
    setPinName('');
  }, [pendingPin, pinName, onSetWaypoint]);

  // relative bearing from userPos to waypoint, corrected by device heading —
  // this is what makes the arrow mean "turn this much", not "north is here"
  const arrowRotation = (() => {
    if (!waypoint || !userPos) return 0;
    const toRad = (d) => (d * Math.PI) / 180;
    const y = Math.sin(toRad(waypoint.lng - userPos.lng)) * Math.cos(toRad(waypoint.lat));
    const x = Math.cos(toRad(userPos.lat)) * Math.sin(toRad(waypoint.lat)) - Math.sin(toRad(userPos.lat)) * Math.cos(toRad(waypoint.lat)) * Math.cos(toRad(waypoint.lng - userPos.lng));
    const bearing = (Math.atan2(y, x) * 180 / Math.PI + 360) % 360;
    return heading != null ? bearing - heading : bearing;
  })();

  return (
    <div className="fixed inset-0 z-50 bg-black/90 flex flex-col">
      <div className="flex items-center justify-between px-5 py-4 bg-stone-950 border-b border-stone-800">
        <h3 className="tracking-widest text-amber-400 text-sm flex items-center gap-2">
          <MapPin className="w-4 h-4" /> {navigating ? `NAVIGATING TO ${waypoint?.name || 'WAYPOINT'}` : "SET WAYPOINT — TAP MAP"}
        </h3>
        <button onClick={onClose} aria-label="Close map" className="text-stone-500 hover:text-amber-300">
          <X className="w-5 h-5" />
        </button>
      </div>

      <div ref={mapDivRef} className="flex-1 relative" />

      {waypoint && userPos && (
        <div className="absolute bottom-24 right-5 bg-black/85 border border-amber-400/60 rounded-full w-16 h-16 flex items-center justify-center">
          <Compass className="w-8 h-8 text-amber-400" style={{ transform: `rotate(${arrowRotation}deg)` }} />
        </div>
      )}

      {pendingPin && (
        <div className="bg-stone-950 border-t border-amber-400/40 p-4 flex items-center gap-3">
          <input
            value={pinName}
            onChange={(e) => setPinName(e.target.value)}
            placeholder="Name this waypoint (e.g. College Gate)"
            className="flex-1 bg-black border border-stone-700 text-stone-200 px-3 py-3 text-sm focus:border-amber-400 outline-none"
          />
          <button onClick={confirmPin} className="bg-amber-400 text-black px-5 py-3 font-bold text-sm tracking-widest">
            DROP PIN
          </button>
          <button onClick={() => setPendingPin(null)} className="border border-stone-700 text-stone-400 px-4 py-3 text-sm">
            CANCEL
          </button>
        </div>
      )}

      {/* Already navigating — offer a clear way to stop the route or just
          close the map and keep walking (route keeps running in the
          background either way; this is just this screen's exit menu). */}
      {navigating && !pendingPin && (
        <div className="bg-stone-950 border-t border-amber-400/40 p-4 flex items-center gap-3">
          <button onClick={onClose} className="flex-1 border border-stone-700 text-stone-300 px-4 py-4 rounded-lg text-sm font-bold tracking-widest">
            CLOSE MAP — KEEP WALKING
          </button>
          <button onClick={onStopNav} className="flex-1 bg-red-950 border border-red-500 text-red-400 px-4 py-4 rounded-lg text-sm font-bold tracking-widest">
            STOP NAVIGATION
          </button>
        </div>
      )}

      {waypoint && !navigating && !pendingPin && (
        <div className="bg-stone-950 border-t border-amber-400/40 p-4 flex items-center justify-between gap-3">
          <span className="text-sm text-stone-300">Target: <strong className="text-amber-400">{waypoint.name}</strong></span>
          <button onClick={onStartNav} className="bg-amber-400 text-black px-5 py-3 font-bold text-sm tracking-widest flex items-center gap-2">
            <Navigation className="w-4 h-4" /> START NAVIGATION
          </button>
        </div>
      )}
    </div>
  );
}