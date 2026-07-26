import { useState, useRef, useCallback, useEffect } from 'react';
import Webcam from 'react-webcam';
import { createWorker } from 'tesseract.js';
import {
  loadDetector, loadPotholeDetector, detectLocal,
  setFocalPx, getFocalPx, generateDescription,
} from './lib/detector';
import { runOpenCvScan, warmOpenCv } from './lib/opencvCrossing';
import emailjs from '@emailjs/browser';
import WaypointMap from './components/WaypointMap';
import AuthModal from './components/AuthModal';
import {
  Camera, Mic, MicOff, History, Bell, AlertTriangle, X, Volume2, VolumeX,
  WifiOff, Settings, ScanText, MapPin, BarChart3, Compass, Navigation, Footprints,
  MessageCircle, TrafficCone, LogIn, LogOut, User,
} from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:5000';

function getDeviceId() {
  let id = localStorage.getItem('visionaid_device_id');
  if (!id) { id = crypto.randomUUID(); localStorage.setItem('visionaid_device_id', id); }
  return id;
}
const DEVICE_ID = getDeviceId();

async function api(path, opts = {}) {
  const token = localStorage.getItem('visionaid_token');
  const authHeader = token ? { authorization: `Bearer ${token}` } : {};
  const res = await fetch(`${API_BASE}${path}`, { ...opts, headers: { 'x-device-id': DEVICE_ID, ...authHeader, ...(opts.headers || {}) } });
  return res.json();
}

function beep(freq = 880, duration = 150) {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.frequency.value = freq;
    osc.connect(gain);
    gain.connect(ctx.destination);
    gain.gain.setValueAtTime(0.15, ctx.currentTime);
    osc.start();
    osc.stop(ctx.currentTime + duration / 1000);
  } catch { /* no-op */ }
}

function haversineM(lat1, lng1, lat2, lng2) {
  const R = 6371000, toRad = (d) => (d * Math.PI) / 180;
  const dLat = toRad(lat2 - lat1), dLng = toRad(lng2 - lng1);
  const a = Math.sin(dLat / 2) ** 2 + Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLng / 2) ** 2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}
function bearingDeg(lat1, lng1, lat2, lng2) {
  const toRad = (d) => (d * Math.PI) / 180;
  const y = Math.sin(toRad(lng2 - lng1)) * Math.cos(toRad(lat2));
  const x = Math.cos(toRad(lat1)) * Math.sin(toRad(lat2)) - Math.sin(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.cos(toRad(lng2 - lng1));
  return (Math.atan2(y, x) * 180 / Math.PI + 360) % 360;
}

// Shrinks a webcam screenshot before it goes anywhere near a vision API.
// This is the real fix for burning through the Groq TPM cap in a few
// requests — image tokens scale with resolution, so a 480px-wide JPEG at
// medium quality costs a fraction of a full 1280px webcam frame.
function resizeScreenshotDataUrl(dataUrl, maxWidth = 400) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const scale = Math.min(1, maxWidth / img.width);
      const canvas = document.createElement('canvas');
      canvas.width = Math.round(img.width * scale);
      canvas.height = Math.round(img.height * scale);
      canvas.getContext('2d').drawImage(img, 0, 0, canvas.width, canvas.height);
      resolve(canvas.toDataURL('image/jpeg', 0.5));
    };
    img.onerror = () => resolve(dataUrl);
    img.src = dataUrl;
  });
}

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const recognitionRef = useRef(null);
  const mediaRecorderRef = useRef(null);

  const [detections, setDetections] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [status, setStatus] = useState("READY");
  const [history, setHistory] = useState([]);
  const [isContinuous, setIsContinuous] = useState(false);
  const [language, setLanguage] = useState("en");
  const [voiceSpeed, setVoiceSpeed] = useState(0.9);
  const [showHistory, setShowHistory] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [showMap, setShowMap] = useState(false);
  const [currentUser, setCurrentUser] = useState(() => {
    try { return JSON.parse(localStorage.getItem('visionaid_user') || 'null'); } catch { return null; }
  });
  const [sosMode, setSosMode] = useState(false);
  const [muted, setMuted] = useState(false);
  const [serverUp, setServerUp] = useState(true);
  const [errorToast, setErrorToast] = useState(null);
  const [isListening, setIsListening] = useState(false);
  const [isReadingText, setIsReadingText] = useState(false);
  const [ocrResult, setOcrResult] = useState(null);
  const [stats, setStats] = useState(null);
  const [detectorReady, setDetectorReady] = useState(false);
  const [potholeReady, setPotholeReady] = useState(false);
  const [calibrating, setCalibrating] = useState(false);
  const [alwaysListening, setAlwaysListening] = useState(false);
  const alwaysListeningRef = useRef(false);
  const lastDescribeAtRef = useRef(0);

  const [isChatting, setIsChatting] = useState(false);
  const isChattingRef = useRef(false);
  const [chatLog, setChatLog] = useState([]);
  const [showChat, setShowChat] = useState(false);

  const [crossingResult, setCrossingResult] = useState(null);
  const [checkingCrossing, setCheckingCrossing] = useState(false);

  const [userPos, setUserPos] = useState(null);
  const [heading, setHeading] = useState(null);
  const [compassGranted, setCompassGranted] = useState(false);

  const [waypoint, setWaypoint] = useState(null);
  const [routeSteps, setRouteSteps] = useState(null);
  const [routePath, setRoutePath] = useState(null);
  const [navStepIndex, setNavStepIndex] = useState(0);
  const [straightLineMode, setStraightLineMode] = useState(false);
  const routeStepsRef = useRef(null);
  const navStepIndexRef = useRef(0);
  const lastNavSpeakRef = useRef(0);

  const lastAlertRef = useRef(0);
  const isLoadingRef = useRef(false);

  const speak = useCallback((text, { urgent = false } = {}) => {
    if (muted) return;
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.lang = language === "hi" ? "hi-IN" : "en-US";
    u.rate = urgent ? Math.min(1.4, voiceSpeed + 0.3) : voiceSpeed;
    u.pitch = urgent ? 1.3 : 1.05;
    window.speechSynthesis.speak(u);
  }, [muted, language, voiceSpeed]);

  const flashError = (msg) => {
    setErrorToast(msg);
    clearTimeout(flashError._t);
    flashError._t = setTimeout(() => setErrorToast(null), 4000);
  };

 useEffect(() => {
  loadDetector()
    .then(() => {
      setDetectorReady(true);

      loadPotholeDetector()
        .then((s) => setPotholeReady(!!s));
    })
    .catch((e) =>
      flashError(
        "On-device model failed to load — " + e.message
      )
    );

  // Start OpenCV in background.
  // The catch prevents a failed warmup from creating
  // an unhandled Promise rejection.
  warmOpenCv().catch(() => {});

  // ...keep EVERYTHING else in this useEffect unchanged

    api('/api/profile').then(({ profile }) => {
      if (!profile) return;
      setLanguage(profile.language || 'en');
      setVoiceSpeed(profile.voiceSpeed ?? 0.9);
      if (profile.focalPx) setFocalPx(profile.focalPx);
    }).catch(() => {});

    fetch(`${API_BASE}/api/health`, { signal: AbortSignal.timeout(3000) }).then(r => setServerUp(r.ok)).catch(() => setServerUp(false));

    if (navigator.geolocation) {
      const watchId = navigator.geolocation.watchPosition(
        (pos) => setUserPos({ lat: pos.coords.latitude, lng: pos.coords.longitude }),
        () => {}, { enableHighAccuracy: true, maximumAge: 2000 }
      );
      return () => navigator.geolocation.clearWatch(watchId);
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const saveProfile = useCallback((patch) => {
    api('/api/profile', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(patch) }).catch(() => {});
  }, []);

  const onOrientation = useCallback((e) => {
    const h = e.webkitCompassHeading ?? (e.absolute ? e.alpha : null);
    if (h != null) setHeading(h);
  }, []);
  const enableCompass = async () => {
    if (typeof DeviceOrientationEvent?.requestPermission === 'function') {
      try {
        const perm = await DeviceOrientationEvent.requestPermission();
        if (perm !== 'granted') return flashError("Compass permission denied");
      } catch { return flashError("Compass permission request failed"); }
    }
    window.addEventListener('deviceorientationabsolute', onOrientation, true);
    window.addEventListener('deviceorientation', onOrientation, true);
    setCompassGranted(true);
  };

  const calibrateDistance = async (distanceM) => {
    if (!webcamRef.current?.video || !detectorReady) return flashError("Model not ready yet");
    setCalibrating(true);
    try {
      const dets = await detectLocal(webcamRef.current.video);
      const person = dets.find(d => d.class === 'person');
      if (!person) return flashError("No person detected — face the camera, stand fully in frame");
      const bboxH = person.bbox[3] - person.bbox[1];
      const newFocalPx = Math.round((bboxH * distanceM) / 1.7);
      setFocalPx(newFocalPx);
      saveProfile({ focalPx: newFocalPx });
      speak(language === 'hi' ? "कैलिब्रेशन पूरा हुआ" : "Calibration complete");
      setStatus(`CALIBRATED // ${newFocalPx}px`);
    } finally {
      setCalibrating(false);
    }
  };

  const EMAILJS_SERVICE_ID = import.meta.env.VITE_EMAILJS_SERVICE_ID;
  const EMAILJS_TEMPLATE_ID = import.meta.env.VITE_EMAILJS_TEMPLATE_ID;
  const EMAILJS_PUBLIC_KEY = import.meta.env.VITE_EMAILJS_PUBLIC_KEY;

  const triggerSOS = useCallback(() => {
    setSosMode(true);
    setStatus("SOS SENDING...");
    speak(language === "hi" ? "आपातकालीन मोड सक्रिय। संपर्कों को सूचित किया जा रहा है।" : "Emergency mode activated. Alerting contacts.", { urgent: true });

    const send = (coords) => {
      const payload = {
        message: 'SOS triggered from VisionAid',
        lat: coords?.latitude ?? userPos?.lat ?? 'unknown',
        lng: coords?.longitude ?? userPos?.lng ?? 'unknown',
        map_link: (coords || userPos) ? `https://maps.google.com/?q=${coords?.latitude ?? userPos.lat},${coords?.longitude ?? userPos.lng}` : 'unavailable',
        description: history[0]?.description || 'No recent scan',
        lang: language,
      };
      if (EMAILJS_SERVICE_ID && EMAILJS_TEMPLATE_ID && EMAILJS_PUBLIC_KEY) {
        emailjs.send(EMAILJS_SERVICE_ID, EMAILJS_TEMPLATE_ID, payload, EMAILJS_PUBLIC_KEY)
          .then(() => setStatus("SOS SENT"))
          .catch(() => { setStatus("SOS EMAIL FAILED"); flashError("EmailJS send failed — check service/template IDs"); })
          .finally(() => setTimeout(() => setSosMode(false), 4000));
      } else {
        fetch(`${API_BASE}/api/sos`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) })
          .then(r => r.json())
          .then(data => { setStatus(data.delivered ? "SOS SENT" : "SOS LOGGED"); })
          .catch(() => setStatus("SOS FAILED"))
          .finally(() => setTimeout(() => setSosMode(false), 5000));
      }
      fetch(`${API_BASE}/api/sos`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }).catch(() => {});
    };
    if (navigator.geolocation) navigator.geolocation.getCurrentPosition((pos) => send(pos.coords), () => send(null), { timeout: 4000 });
    else send(null);
  }, [language, speak, userPos, history]);

  const drawOverlay = useCallback((dets, videoEl) => {
    const canvas = canvasRef.current;
    if (!canvas || !videoEl) return;
    const { videoWidth, videoHeight } = videoEl;
    if (!videoWidth) return;
    canvas.width = videoWidth;
    canvas.height = videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    dets.forEach((d) => {
      const [x1, y1, x2, y2] = d.bbox;
      const color = d.proximity === 'very close' ? '#ef4444' : d.proximity === 'near' ? '#f59e0b' : '#22c55e';
      ctx.strokeStyle = color;
      ctx.lineWidth = d.outdoorPriority ? 4 : 2;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      const label = `${d.class} ${d.distance_m}${d.approximate ? '~' : ''}m`;
      ctx.font = 'bold 18px system-ui';
      const padding = 6;
      const textW = ctx.measureText(label).width;
      ctx.fillStyle = color;
      ctx.fillRect(x1, Math.max(0, y1 - 26), textW + padding * 2, 26);
      ctx.fillStyle = '#000';
      ctx.fillText(label, x1 + padding, Math.max(18, y1 - 7));
    });
  }, []);

  const detectLocalTick = useCallback(async () => {
    if (!webcamRef.current?.video || webcamRef.current.video.readyState < 2) return;
    try {
      const dets = await detectLocal(webcamRef.current.video);
      setDetections(dets);
      drawOverlay(dets, webcamRef.current.video);
      const closest = dets[0];
      const now = Date.now();
      if (closest && closest.proximity === 'very close' && now - lastAlertRef.current > 3000) {
        lastAlertRef.current = now;
        beep(1000, 200);
        if (navigator.vibrate) navigator.vibrate([120, 60, 120]);
        speak(language === "hi" ? `सावधान! ${closest.class} बहुत पास है।` : `Warning! ${closest.class} very close.`, { urgent: true });
      }
      setStatus(`LIVE // ${dets.length} DETECTED`);
    } catch (e) {
      console.warn('Local tick failed:', e.message);
    }
  }, [drawOverlay, speak, language]);

  const detectObjects = useCallback(async () => {
    if (!webcamRef.current || isLoadingRef.current) return;
    isLoadingRef.current = true;
    setIsLoading(true);
    setStatus("ANALYZING...");
    try {
      if (detectorReady && webcamRef.current.video?.readyState >= 2) {
        const dets = await detectLocal(webcamRef.current.video);
        const description = generateDescription(dets, language);
        setDetections(dets);
        drawOverlay(dets, webcamRef.current.video);
        const entry = { time: new Date().toLocaleTimeString(), detections: dets, description };
        setHistory(prev => [entry, ...prev].slice(0, 12));
        const closest = dets[0];
        const now = Date.now();
        if (closest && closest.proximity === 'very close' && now - lastAlertRef.current > 3000) {
          lastAlertRef.current = now;
          beep(1000, 200);
          if (navigator.vibrate) navigator.vibrate([120, 60, 120]);
          speak(language === "hi" ? `सावधान! ${closest.class} बहुत पास है, लगभग ${closest.distance_m} मीटर।` : `Warning! ${closest.class} very close, about ${closest.distance_m} meters.`, { urgent: true });
        } else {
          speak(description);
        }
        setStatus(`SCAN COMPLETE // ${dets.length} DETECTED`);
        api('/api/history', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ description, count: dets.length, closest_distance_m: closest?.distance_m ?? null, classes: dets.map(d => d.class) }) }).catch(() => {});
        return;
      }
      setStatus("MODEL LOADING — TRY AGAIN IN A MOMENT");
    } catch (e) {
      setStatus("ERROR");
      flashError("Detection failed: " + e.message);
    } finally {
      isLoadingRef.current = false;
      setIsLoading(false);
    }
  }, [drawOverlay, speak, language, detectorReady]);

  useEffect(() => {
    if (!isContinuous) return;
    let cancelled = false, timeoutId;
    const loop = async () => {
      if (cancelled) return;
      await detectLocalTick();
      if (!cancelled) timeoutId = setTimeout(loop, 300);
    };
    loop();
    return () => { cancelled = true; clearTimeout(timeoutId); };
  }, [isContinuous, detectLocalTick]);

  const runOCR = useCallback(async () => {
    if (!webcamRef.current) return;
    setIsReadingText(true);
    setStatus("READING TEXT...");
    speak(language === "hi" ? "पाठ पढ़ रहा हूँ..." : "Reading text...");
    try {
      const screenshot = webcamRef.current.getScreenshot();
      const worker = await createWorker(language === "hi" ? "hin" : "eng");
      const { data } = await worker.recognize(screenshot);
      await worker.terminate();
      const text = (data.text || "").trim();
      setOcrResult(text);
      if (text) { speak(text); setStatus("TEXT READ COMPLETE"); }
      else { speak(language === "hi" ? "कोई पाठ नहीं मिला" : "No text found"); setStatus("NO TEXT FOUND"); }
    } catch (err) {
      flashError("OCR failed: " + err.message);
      setStatus("OCR FAILED");
    } finally {
      setIsReadingText(false);
    }
  }, [language, speak]);

  const describeScene = useCallback(async () => {
    if (!webcamRef.current) return;
    const waited = Date.now() - lastDescribeAtRef.current;
    if (waited < 4000) return flashError(`Wait ${Math.ceil((4000 - waited) / 1000)}s before asking again`);
    lastDescribeAtRef.current = Date.now();

    setIsLoading(true);
    setStatus(language === 'hi' ? "दृश्य समझ रहा हूँ..." : "UNDERSTANDING SCENE...");
    try {
      const raw = webcamRef.current.getScreenshot();
      if (!raw) return;
      const resized = await resizeScreenshotDataUrl(raw, 480);
      const res = await fetch(resized);
      const blob = await res.blob();
      const formData = new FormData();
      formData.append('image', blob, 'capture.jpg');
      formData.append('lang', language);
      formData.append('fallback_description', generateDescription(detections, language));

      const response = await fetch(`${API_BASE}/api/describe-scene`, { method: 'POST', headers: { 'x-device-id': DEVICE_ID }, body: formData });
      const data = await response.json();
      if (data.success) {
        speak(data.description);
        setStatus(data.provider === 'on-device-fallback' ? "DESCRIBED (ON-DEVICE FALLBACK)" : "SCENE DESCRIBED");
      } else {
        flashError(data.error || "Could not describe scene");
        setStatus("DESCRIBE FAILED");
      }
    } catch {
      flashError("Could not reach description service");
    } finally {
      setIsLoading(false);
    }
  }, [language, speak, detections]);

  const checkForCrossing = useCallback(async () => {
    if (!webcamRef.current?.video) return;
    setCheckingCrossing(true);
    setStatus("RUNNING OPENCV SCAN...");
    try {
      const result = await runOpenCvScan(webcamRef.current.video);
      setCrossingResult(result);
      const { crossing: c, puddle: p, lighting: l } = result;

      const lines = [];
      if (c) lines.push(c.detected
        ? (language === 'hi' ? `ज़ेब्रा क्रॉसिंग मिली, ${c.stripeCount} पट्टियाँ।` : `Crossing detected, ${c.stripeCount} stripes visible.`)
        : (language === 'hi' ? "कोई क्रॉसिंग नहीं मिली।" : "No crossing here."));
      if (p?.detected) lines.push(language === 'hi' ? "सामने गीली/चमकदार सतह हो सकती है।" : "Possible wet or reflective surface ahead.");
      if (l?.lowLight) lines.push(language === 'hi' ? "रोशनी कम है, सावधानी बरतें।" : "Low light — proceed with extra caution.");

      speak(lines.join(' ') || (language === 'hi' ? "जांच पूरी हुई" : "Scan complete."));
      setStatus("OPENCV SCAN DONE");
    } catch (e) {
      // worker/CDN genuinely unreachable — say so plainly instead of hanging
      flashError("OpenCV scan failed: " + e.message);
      speak(language === 'hi' ? "जांच अभी उपलब्ध नहीं है" : "Scan not available right now.");
      setCrossingResult(null);
    } finally {
      setCheckingCrossing(false);
    }
  }, [language, speak]);

  const askAI = useCallback(async () => {
    if (isChatting) return;
    if (!navigator.mediaDevices?.getUserMedia) return flashError("Microphone not available for AI chat");
    isChattingRef.current = true;
    setIsChatting(true);
    // release the mic from always-listening/voice-command mode first — two
    // things trying to use the microphone at once was part of what made
    // voice commands flaky
    recognitionRef.current?.stop();
    setIsListening(false);
    setStatus(language === 'hi' ? "सुन रहा हूँ..." : "LISTENING FOR YOUR QUESTION...");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const chunks = [];
      const recorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
      mediaRecorderRef.current = recorder;
      recorder.ondataavailable = (e) => chunks.push(e.data);
      recorder.start();
      speak(language === 'hi' ? "पूछिए..." : "Go ahead, ask.");
      await new Promise((resolve) => setTimeout(resolve, 4500));
      recorder.stop();
      await new Promise((resolve) => { recorder.onstop = resolve; });
      stream.getTracks().forEach(t => t.stop());

      const audioBlob = new Blob(chunks, { type: 'audio/webm' });
      const form = new FormData();
      form.append('audio', audioBlob, 'question.webm');
      form.append('lang', language);
      setStatus("TRANSCRIBING...");
      const tRes = await fetch(`${API_BASE}/api/transcribe`, { method: 'POST', headers: { 'x-device-id': DEVICE_ID }, body: form });
      const tData = await tRes.json();
      if (!tData.success || !tData.text) { flashError(tData.error || "Didn't catch that — try again"); return; }

      setStatus("THINKING...");
      const context = history[0]?.description || generateDescription(detections, language);
      const cRes = await api('/api/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question: tData.text, context, lang: language }) });
      if (!cRes.success) { flashError(cRes.error || "Chat failed"); return; }

      setChatLog(prev => [{ q: tData.text, a: cRes.answer }, ...prev].slice(0, 10));
      speak(cRes.answer);
      setStatus("READY");
    } catch (e) {
      flashError("AI chat failed: " + e.message);
    } finally {
      isChattingRef.current = false;
      setIsChatting(false);
      // hand the mic back to always-listening mode if that was on
      if (alwaysListeningRef.current && !isListening) setTimeout(() => toggleVoiceCommand(), 500);
    }
  }, [language, speak, history, detections, isChatting, isListening]);

  const setWaypointAndClose = (wp) => setWaypoint(wp);

  const startNavigation = async () => {
    if (!waypoint) return flashError("Set a waypoint first");
    if (!userPos) return flashError("Waiting for GPS fix — try again in a moment");
    if (!compassGranted) await enableCompass();
    try {
      const data = await api('/api/directions', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ originLat: userPos.lat, originLng: userPos.lng, destLat: waypoint.lat, destLng: waypoint.lng }),
      });
      if (!data.success) {
        if (data.straight_line_available) {
          setStraightLineMode(true);
          routeStepsRef.current = [{ instruction: 'Head toward waypoint', end: waypoint, distance_text: '' }];
          navStepIndexRef.current = 0;
          setRouteSteps(routeStepsRef.current);
          setRoutePath(null);
          setNavStepIndex(0);
          setShowMap(false);
          speak(language === 'hi' ? "सीधी दिशा नेविगेशन शुरू।" : "Straight-line navigation started — no road routing available.");
          return;
        }
        return flashError(data.error || "Could not get directions");
      }
      setStraightLineMode(false);
      routeStepsRef.current = data.steps;
      navStepIndexRef.current = 0;
      setRouteSteps(data.steps);
      setRoutePath(data.path || null);
      setNavStepIndex(0);
      setShowMap(false);
      speak(`${language === 'hi' ? 'नेविगेशन शुरू।' : 'Navigation started.'} ${data.steps[0].instruction}, ${data.steps[0].distance_text}.`);
    } catch {
      flashError("Directions request failed");
    }
  };

  const stopNavigation = () => {
    routeStepsRef.current = null;
    navStepIndexRef.current = 0;
    setRouteSteps(null);
    setRoutePath(null);
    setWaypoint(null);
    setNavStepIndex(0);
    setStraightLineMode(false);
  };

  const relativeDirection = (targetBearing) => {
    if (heading == null) return language === 'hi' ? 'दिशा सेंसर उपलब्ध नहीं' : 'compass unavailable';
    let diff = (targetBearing - heading + 360) % 360;
    if (diff > 180) diff -= 360;
    const abs = Math.abs(diff);
    if (abs <= 15) return language === 'hi' ? 'सीधे आगे बढ़ें' : 'go straight ahead';
    if (abs <= 60) return diff > 0 ? (language === 'hi' ? 'थोड़ा दाईं ओर मुड़ें' : 'turn slightly right') : (language === 'hi' ? 'थोड़ा बाईं ओर मुड़ें' : 'turn slightly left');
    if (abs <= 135) return diff > 0 ? (language === 'hi' ? 'दाईं ओर मुड़ें' : 'turn right') : (language === 'hi' ? 'बाईं ओर मुड़ें' : 'turn left');
    return language === 'hi' ? 'पीछे मुड़ें' : 'turn around';
  };

  useEffect(() => {
    if (!userPos || !routeStepsRef.current) return;
    const steps = routeStepsRef.current;
    const idx = navStepIndexRef.current;
    const step = steps[idx];
    if (!step) return;
    const dist = haversineM(userPos.lat, userPos.lng, step.end.lat, step.end.lng);
    const bearing = bearingDeg(userPos.lat, userPos.lng, step.end.lat, step.end.lng);
    const arriveThreshold = straightLineMode ? 10 : 20;

    if (dist < arriveThreshold) {
      const nextIdx = idx + 1;
      if (nextIdx >= steps.length) {
        speak(language === "hi" ? "आप अपने गंतव्य पर पहुँच गए हैं।" : "You have arrived at your destination.");
        stopNavigation();
      } else {
        navStepIndexRef.current = nextIdx;
        setNavStepIndex(nextIdx);
        speak(`${steps[nextIdx].instruction}, ${steps[nextIdx].distance_text}.`);
      }
    } else {
      const now = Date.now();
      if (now - lastNavSpeakRef.current > (straightLineMode ? 6000 : 8000)) {
        lastNavSpeakRef.current = now;
        speak(`${relativeDirection(bearing)}, ${Math.round(dist)}${language === 'hi' ? ' मीटर' : 'm'}.`);
      }
    }
  }, [userPos]); // eslint-disable-line react-hooks/exhaustive-deps

  // While actively navigating, periodically describe the scene without
  // being asked — this is the "check the road every so often" behavior.
  // Skipped if something else is already using the camera/AI pipeline.
  useEffect(() => {
    if (!routeSteps) return;
    const interval = setInterval(() => {
      if (!isLoading && !isReadingText && !isChatting && !checkingCrossing) {
        describeScene();
      }
    }, 9000);
    return () => clearInterval(interval);
  }, [routeSteps, isLoading, isReadingText, isChatting, checkingCrossing, describeScene]);

  const regexFallbackDispatch = (heard) => {
    if (/front|ahead|सामने|आगे/.test(heard)) detectObjects();
    else if (/cross(ing)?|zebra|पार|क्रॉसिंग/.test(heard)) checkForCrossing();
    else if (/describe|what.*see|scene|वर्णन|दृश्य/.test(heard)) describeScene();
    else if (/read|text|sign|पढ़ो|पाठ/.test(heard)) runOCR();
    else if (/help|emergency|sos|मदद|आपातकाल/.test(heard)) triggerSOS();
    else if (/mute|silence|चुप/.test(heard)) { setMuted(true); window.speechSynthesis.cancel(); }
    else if (/unmute|आवाज़/.test(heard)) setMuted(false);
    else if (/setting/.test(heard)) setShowSettings(true);
    else if (/map|waypoint|navigate/.test(heard)) setShowMap(true);
    else if (/stat/.test(heard)) openStats();
    else askAI();
  };

  const dispatchVoiceCommand = async (heard) => {
    try {
      const ac = new AbortController();
      const timeout = setTimeout(() => ac.abort(), 2000);
      const res = await fetch(`${API_BASE}/api/voice-intent`, {
        method: 'POST', headers: { 'Content-Type': 'application/json', 'x-device-id': DEVICE_ID },
        body: JSON.stringify({ transcript: heard, lang: language }), signal: ac.signal,
      });
      clearTimeout(timeout);
      const data = await res.json();
      if (!data.success) throw new Error('intent parse failed');
      switch (data.intent) {
        case 'analyze': detectObjects(); break;
        case 'describe': describeScene(); break;
        case 'read_text': runOCR(); break;
        case 'sos': triggerSOS(); break;
        case 'crossing': checkForCrossing(); break;
        case 'mute': setMuted(true); window.speechSynthesis.cancel(); break;
        case 'unmute': setMuted(false); break;
        case 'settings': setShowSettings(true); break;
        case 'stats': openStats(); break;
        case 'history': setShowHistory(true); break;
        case 'map': case 'navigate': setShowMap(true); break;
        case 'chat': askAI(); break;
        default: regexFallbackDispatch(heard);
      }
    } catch {
      regexFallbackDispatch(heard);
    }
  };

  const toggleVoiceCommand = () => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) return flashError("Voice recognition not supported in this browser — try Chrome");
    if (location.protocol !== 'https:' && location.hostname !== 'localhost' && location.hostname !== '127.0.0.1') {
      return flashError("Voice commands need HTTPS or localhost");
    }
    if (isListening) {
      // manual stop = really stop, not "stop for one cycle then auto-restart"
      alwaysListeningRef.current = false;
      setAlwaysListening(false);
      recognitionRef.current?.stop();
      setIsListening(false);
      return;
    }
    if (isChattingRef.current) return; // mic is busy with AI chat right now

    const recognition = new SpeechRecognition();
    recognition.lang = language === "hi" ? "hi-IN" : "en-US";
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.onstart = () => { setIsListening(true); setStatus("LISTENING..."); };
    recognition.onend = () => {
      setIsListening(false);
      // don't restart into a busy mic, and don't restart if the user (or
      // askAI) has since turned always-listening off
      if (alwaysListeningRef.current && !isChattingRef.current) setTimeout(() => toggleVoiceCommand(), 400);
    };
    recognition.onerror = (event) => {
      setIsListening(false);
      const reasons = {
        'no-speech': null, 'not-allowed': "Microphone permission denied", 'service-not-allowed': "Microphone permission denied",
        'audio-capture': "No microphone found", 'network': "Voice recognition needs internet", 'aborted': null,
      };
      const msg = reasons[event.error];
      if (msg === undefined) flashError(`Voice recognition error: ${event.error}`);
      else if (msg) flashError(msg);
    };
    recognition.onresult = (event) => {
      const heard = event.results[0][0].transcript.toLowerCase();
      setStatus(`HEARD: "${heard}"`);
      dispatchVoiceCommand(heard);
    };
    recognitionRef.current = recognition;
    recognition.start();
  };

  const openStats = () => { api('/api/stats').then(setStats).catch(() => flashError("Could not load stats")); };
  const proximityColor = (p) => p === 'very close' ? 'text-red-400 border-red-500/60' : p === 'near' ? 'text-amber-400 border-amber-400/60' : 'text-emerald-400 border-emerald-500/60';

  // Mandatory login gate — the app does nothing useful without an account,
  // since history/settings/locations are all tied to a real user now, not
  // just an anonymous browser device-id. No skip option.
  if (!currentUser) {
  return (
    <div className="min-h-screen bg-[#12130f] text-stone-200 font-sans flex items-center justify-center p-4">
      
      <div className="w-full max-w-sm flex flex-col items-center gap-8">
        
        {/* VisionAid branding */}
        <div className="text-center">
          <div className="w-14 h-14 rounded-xl bg-amber-400 flex items-center justify-center mx-auto mb-3">
            <Footprints className="w-8 h-8 text-black" />
          </div>

          <h1 className="text-3xl font-bold text-amber-400">
            VisionAid
          </h1>

          <p className="text-stone-500 text-sm mt-2 leading-relaxed">
            Log in or create an account to continue — your scans,
            settings, and locations are saved to your account.
          </p>
        </div>

        {/* Login / Signup form */}
        <AuthModal
          embedded
          onClose={() => {}}
          onAuthed={(user) => {
            setCurrentUser(user);

            api('/api/profile').then(({ profile }) => {
              if (profile) {
                setLanguage(profile.language || 'en');
                setVoiceSpeed(profile.voiceSpeed ?? 0.9);

                if (profile.focalPx) {
                  setFocalPx(profile.focalPx);
                }
              }
            });
          }}
        />

      </div>
    </div>
  );
}

  return (
    <div className="min-h-screen bg-[#12130f] text-stone-200 font-sans">
      <div aria-live="assertive" className="sr-only">{status}</div>
      <style>{`
        @keyframes scanline { 0% { transform: translateY(-100%); } 100% { transform: translateY(2000%); } }
        @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.2; } }
        @keyframes slideIn { from { transform: translateY(-12px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }
        @keyframes sosPulse { 0%, 100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.6); } 50% { box-shadow: 0 0 0 14px rgba(239,68,68,0); } }
        @keyframes listenPulse { 0%, 100% { box-shadow: 0 0 0 0 rgba(251,191,36,0.5); } 50% { box-shadow: 0 0 0 12px rgba(251,191,36,0); } }
        .scan-line { animation: scanline 1.6s linear infinite; }
        .blink-dot { animation: blink 1.4s ease-in-out infinite; }
        .toast-in { animation: slideIn 0.2s ease-out; }
        .sos-pulse { animation: sosPulse 1s ease-out infinite; }
        .listen-pulse { animation: listenPulse 1s ease-out infinite; }
        button:focus-visible, [tabindex]:focus-visible { outline: 3px solid #f59e0b; outline-offset: 2px; }
      `}</style>

      {errorToast && (
        <div className="fixed top-4 left-1/2 -translate-x-1/2 z-[60] toast-in">
          <div className="bg-red-950 border border-red-500 text-red-300 px-5 py-3 text-sm flex items-center gap-3 shadow-lg rounded-lg">
            <AlertTriangle className="w-4 h-4 shrink-0" />{errorToast}
          </div>
        </div>
      )}

      {routeSteps && (
        <div className="fixed top-0 left-0 right-0 bg-amber-400 text-black text-center py-2.5 text-sm font-bold z-[60] flex items-center justify-center gap-4">
          <Footprints className="w-4 h-4" />
          <span>{straightLineMode ? 'STRAIGHT-LINE NAV: ' : `STEP ${navStepIndex + 1}/${routeSteps.length}: `}{routeSteps[navStepIndex]?.instruction} {routeSteps[navStepIndex]?.distance_text}</span>
          <button onClick={stopNavigation} className="underline">STOP</button>
        </div>
      )}

      {sosMode && (
        <div className="fixed top-0 left-0 right-0 bg-red-600 text-black text-center py-2.5 text-sm font-bold z-[60]">
          🚨 SOS ACTIVE — ALERTING EMERGENCY CONTACTS
        </div>
      )}

      <div className="max-w-6xl mx-auto px-5 py-8">
        <div className="flex flex-wrap justify-between items-start gap-4 mb-6 border-b border-stone-800 pb-6">
          <div>
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-amber-400 flex items-center justify-center">
                <Footprints className="w-6 h-6 text-black" />
              </div>
              <h1 className="text-4xl sm:text-5xl font-bold tracking-tight text-amber-400">VisionAid</h1>
            </div>
            <p className="text-stone-500 text-xs sm:text-sm mt-2">Outdoor navigation assistant — obstacles, traffic, crossings, GPS guidance, AI chat</p>
          </div>
          <div className="flex items-center gap-2 flex-wrap">
            <button onClick={() => setShowSettings(true)} aria-label="Settings" className="p-2.5 border border-stone-700 rounded-lg text-stone-300 hover:border-amber-400 hover:text-amber-300 transition-colors"><Settings className="w-4 h-4" /></button>
            <button onClick={() => setShowMap(true)} aria-label="Waypoint map" className="p-2.5 border border-stone-700 rounded-lg text-stone-300 hover:border-amber-400 hover:text-amber-300 transition-colors"><MapPin className="w-4 h-4" /></button>
            <button onClick={openStats} aria-label="Usage stats" className="p-2.5 border border-stone-700 rounded-lg text-stone-300 hover:border-amber-400 hover:text-amber-300 transition-colors"><BarChart3 className="w-4 h-4" /></button>
            <button
              onClick={() => { localStorage.removeItem('visionaid_token'); localStorage.removeItem('visionaid_user'); setCurrentUser(null); }}
              className="flex items-center gap-2 px-3 py-2.5 rounded-lg border border-stone-700 text-stone-300 hover:border-red-400 hover:text-red-300 text-xs transition-colors"
            >
              <User className="w-4 h-4" /> {currentUser.name || currentUser.email.split('@')[0]} <LogOut className="w-3.5 h-3.5" />
            </button>
            <button onClick={triggerSOS} aria-label="Trigger emergency SOS" className={`flex items-center gap-2 px-5 py-2.5 rounded-lg border font-bold text-xs transition-colors ${sosMode ? 'bg-red-600 border-red-600 text-black sos-pulse' : 'bg-red-950 border-red-500 text-red-400 hover:bg-red-900'}`}>
              <Bell className="w-4 h-4" /> SOS
            </button>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-x-6 gap-y-2 mb-4 text-xs sm:text-sm">
          <div className="flex items-center gap-2">
            {!serverUp ? <WifiOff className="w-4 h-4 text-red-400" /> : <span className={`w-2 h-2 rounded-full ${isLoading ? 'bg-amber-400 blink-dot' : 'bg-emerald-500'}`} />}
            <span className={!serverUp ? 'text-red-400' : 'text-stone-300'}>{!serverUp ? 'SERVER OFFLINE' : status}</span>
          </div>
          <span className={detectorReady ? 'text-emerald-500' : 'text-stone-600'}>{detectorReady ? '● VISION MODEL READY' : '○ LOADING VISION MODEL...'}</span>
          <span className={potholeReady ? 'text-emerald-500' : 'text-stone-600'}>{potholeReady ? '● ROAD-HAZARD MODEL ON' : '○ ROAD-HAZARD MODEL OFF (optional)'}</span>
          <span className={userPos ? 'text-sky-400 flex items-center gap-1' : 'text-stone-600 flex items-center gap-1'}><MapPin className="w-3.5 h-3.5" /> {userPos ? `GPS FIXED` : 'GPS SEARCHING...'}</span>
          <button onClick={enableCompass} className={`flex items-center gap-1 ${compassGranted ? 'text-sky-400' : 'text-stone-500 underline'}`}>
            <Compass className="w-3.5 h-3.5" /> {compassGranted ? `COMPASS ${heading != null ? Math.round(heading) + '°' : 'WAITING'}` : 'ENABLE COMPASS'}
          </button>
          <label className="flex items-center gap-2 text-stone-400">
            <input type="checkbox" checked={alwaysListening} onChange={(e) => { const v = e.target.checked; setAlwaysListening(v); alwaysListeningRef.current = v; if (v && !isListening) toggleVoiceCommand(); }} className="accent-amber-400" />
            ALWAYS LISTENING
          </label>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          <div className="lg:col-span-8">
            <div className="relative border border-stone-800 bg-stone-950 overflow-hidden rounded-2xl">
              <Webcam ref={webcamRef} className="w-full block" videoConstraints={{ facingMode: "environment" }} />
              <canvas ref={canvasRef} className="absolute inset-0 w-full h-full pointer-events-none" />
              {(isListening || isChatting) && (
                <div className="absolute bottom-4 left-1/2 -translate-x-1/2 bg-black/80 border border-amber-400 px-4 py-2 rounded-full flex items-center gap-2 listen-pulse">
                  <Mic className="w-4 h-4 text-amber-400" /><span className="text-xs text-amber-300">{isChatting ? "AI CHAT — LISTENING" : "LISTENING"}</span>
                </div>
              )}
              {(isLoading || isReadingText || checkingCrossing) && (
                <>
                  <div className="absolute inset-0 bg-black/40" />
                  <div className="absolute left-0 right-0 h-1 bg-amber-400/70 shadow-[0_0_12px_2px_rgba(245,158,11,0.7)] scan-line" />
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="bg-black/80 border border-amber-400/40 px-6 py-4 rounded-xl text-center">
                      <div className="w-10 h-10 border-2 border-amber-400 border-t-transparent rounded-full animate-spin mx-auto mb-3" />
                      <p className="text-sm text-amber-300">{isReadingText ? "READING TEXT" : checkingCrossing ? "SCANNING FOR CROSSING" : "PROCESSING FRAME"}</p>
                    </div>
                  </div>
                </>
              )}
            </div>

            {ocrResult !== null && (
              <div className="mt-3 border border-stone-800 bg-stone-950/60 p-4 rounded-xl text-sm text-stone-300 flex items-start gap-3">
                <ScanText className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" /><span>{ocrResult || (language === "hi" ? "कोई पाठ नहीं मिला" : "No text found")}</span>
              </div>
            )}
            {crossingResult && (
              <div className="mt-3 space-y-2">
                {crossingResult.crossing && (
                  <div className={`border p-4 rounded-xl text-sm flex items-start gap-3 ${crossingResult.crossing.detected ? 'border-emerald-600 bg-emerald-950/40 text-emerald-300' : 'border-stone-800 bg-stone-950/60 text-stone-400'}`}>
                    <TrafficCone className="w-4 h-4 shrink-0 mt-0.5" />
                    <span>{crossingResult.crossing.detected ? `Zebra crossing likely detected — ${crossingResult.crossing.stripeCount} parallel stripes found.` : 'No crossing pattern detected in current view.'}</span>
                  </div>
                )}
                {crossingResult.puddle?.detected && (
                  <div className="border border-sky-700 bg-sky-950/40 text-sky-300 p-4 rounded-xl text-sm flex items-start gap-3">
                    <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
                    <span>Possible wet/reflective surface ahead ({crossingResult.puddle.coveragePct}% of lower frame).</span>
                  </div>
                )}
                {crossingResult.lighting?.lowLight && (
                  <div className="border border-amber-700 bg-amber-950/40 text-amber-300 p-4 rounded-xl text-sm flex items-start gap-3">
                    <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
                    <span>Low light detected (brightness {crossingResult.lighting.brightness}/255) — proceed carefully.</span>
                  </div>
                )}
              </div>
            )}
          </div>

          <div className="lg:col-span-4 flex flex-col gap-3">
            <button onClick={detectObjects} disabled={isLoading} className="bg-amber-400 hover:bg-amber-300 disabled:opacity-40 text-black py-8 text-2xl font-bold rounded-2xl flex items-center justify-center gap-4 transition-colors active:scale-[0.98]">
              <Camera className="w-8 h-8" strokeWidth={2.5} /> ANALYZE
            </button>

            <div className="grid grid-cols-3 gap-3">
              <button onClick={() => setIsContinuous(!isContinuous)} className={`py-6 text-sm font-bold rounded-xl border transition-colors active:scale-[0.98] ${isContinuous ? 'bg-red-950 border-red-500 text-red-400' : 'bg-stone-950 border-stone-700 text-stone-300 hover:border-amber-400 hover:text-amber-300'}`}>
                {isContinuous ? "LIVE • STOP" : "LIVE MODE"}
              </button>
              <button onClick={describeScene} disabled={isLoading} className="py-6 text-sm font-bold rounded-xl border border-stone-700 bg-stone-950 text-stone-300 hover:border-amber-400 hover:text-amber-300 disabled:opacity-40 transition-colors active:scale-[0.98]">
                DESCRIBE (AI)
              </button>
              <button onClick={runOCR} disabled={isReadingText} className="py-6 text-sm font-bold rounded-xl border border-stone-700 bg-stone-950 text-stone-300 hover:border-amber-400 hover:text-amber-300 disabled:opacity-40 transition-colors active:scale-[0.98] flex items-center justify-center gap-2">
                <ScanText className="w-5 h-5" /> READ TEXT
              </button>
            </div>

            <button onClick={checkForCrossing} disabled={checkingCrossing} className="py-6 text-base font-bold rounded-xl border border-emerald-700 bg-emerald-950/30 text-emerald-300 hover:border-emerald-400 disabled:opacity-40 transition-colors active:scale-[0.98] flex items-center justify-center gap-3">
              <TrafficCone className="w-5 h-5" /> OPENCV SCAN (CROSSING / WET / LIGHT)
            </button>

            <button onClick={() => setShowMap(true)} className="py-6 text-base font-bold rounded-xl border border-sky-700 bg-sky-950/40 text-sky-300 hover:border-sky-400 transition-colors active:scale-[0.98] flex items-center justify-center gap-3">
              <Navigation className="w-5 h-5" /> {waypoint ? `NAVIGATING: ${waypoint.name}` : "SET WAYPOINT & NAVIGATE"}
            </button>

            <button onClick={askAI} disabled={isChatting} className="py-6 text-base font-bold rounded-xl border border-violet-700 bg-violet-950/30 text-violet-300 hover:border-violet-400 disabled:opacity-40 transition-colors active:scale-[0.98] flex items-center justify-center gap-3">
              <MessageCircle className="w-5 h-5" /> {isChatting ? "LISTENING..." : "ASK AI ANYTHING"}
            </button>

            <button onClick={toggleVoiceCommand} className={`py-6 text-base font-bold rounded-xl border flex items-center justify-center gap-3 transition-colors active:scale-[0.98] ${isListening ? 'bg-amber-400 border-amber-400 text-black' : 'bg-stone-950 border-stone-700 text-stone-300 hover:border-amber-400 hover:text-amber-300'}`}>
              {isListening ? <MicOff className="w-5 h-5" /> : <Mic className="w-5 h-5" />}
              {isListening ? "STOP LISTENING" : "VOICE COMMAND"}
            </button>

            <div className="grid grid-cols-3 gap-3">
              <button onClick={() => setShowHistory(true)} disabled={history.length === 0} className="py-4 rounded-xl border border-stone-800 text-stone-500 hover:text-amber-300 hover:border-amber-400/50 disabled:opacity-30 flex items-center justify-center gap-2 text-sm transition-colors"><History className="w-4 h-4" /> LOG</button>
              <button onClick={() => setShowChat(true)} disabled={chatLog.length === 0} className="py-4 rounded-xl border border-stone-800 text-stone-500 hover:text-violet-300 hover:border-violet-400/50 disabled:opacity-30 flex items-center justify-center gap-2 text-sm transition-colors"><MessageCircle className="w-4 h-4" /> CHAT</button>
              <button onClick={() => { setMuted(m => !m); window.speechSynthesis.cancel(); }} className={`py-4 rounded-xl border flex items-center justify-center gap-2 text-sm transition-colors ${muted ? 'border-red-500/50 text-red-400 hover:bg-red-950' : 'border-stone-800 text-stone-500 hover:text-amber-300 hover:border-amber-400/50'}`}>
                {muted ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
              </button>
            </div>
          </div>
        </div>

        {detections.length > 0 && (
          <div className="mt-8 border border-stone-800 bg-stone-950/60 p-6 sm:p-8 rounded-2xl">
            <h3 className="text-lg mb-6 flex items-center gap-3 text-amber-400 font-bold"><AlertTriangle className="w-5 h-5" /> DETECTED — CLOSEST & OUTDOOR-PRIORITY FIRST</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {detections.map((item, i) => (
                <div key={i} className={`border bg-black p-5 rounded-xl transition-colors ${proximityColor(item.proximity).split(' ')[1]}`}>
                  <div className="text-2xl capitalize font-bold mb-1 text-stone-100">{item.class}{item.approximate ? '*' : ''}</div>
                  <div className="text-amber-400 text-lg font-bold mb-2">~{item.distance_m}m</div>
                  <div className="flex items-center gap-2 text-xs mb-3 text-stone-500 uppercase"><span>{item.position}</span><span>•</span><span className={proximityColor(item.proximity).split(' ')[0]}>{item.proximity}</span></div>
                  <div className="h-1 bg-stone-800 mb-2 rounded-full overflow-hidden"><div className="h-1 bg-amber-400" style={{ width: `${Math.round(item.confidence * 100)}%` }} /></div>
                  <div className="text-stone-500 text-xs">{(item.confidence * 100).toFixed(0)}% CONF</div>
                </div>
              ))}
            </div>
          </div>
        )}
        {detections.length === 0 && !isLoading && (
          <div className="mt-8 border border-dashed border-stone-800 rounded-2xl py-10 text-center text-stone-600 text-sm">NO SCAN DATA — PRESS ANALYZE OR SAY "WHAT'S IN FRONT OF ME"</div>
        )}

        {showHistory && (
          <Modal onClose={() => setShowHistory(false)} title="SCAN LOG">
            {history.length === 0 && <p className="text-stone-600 text-sm">No scans recorded yet.</p>}
            {history.map((entry, i) => (
              <div key={i} className="border-l-2 border-amber-400/50 pl-4 mb-4">
                <div className="text-xs text-stone-500 mb-1">{entry.time}</div>
                <div className="text-sm text-stone-300">{entry.description}</div>
              </div>
            ))}
          </Modal>
        )}

        {showChat && (
          <Modal onClose={() => setShowChat(false)} title="AI CHAT HISTORY">
            {chatLog.length === 0 && <p className="text-stone-600 text-sm">No conversations yet — tap "ASK AI ANYTHING".</p>}
            {chatLog.map((c, i) => (
              <div key={i} className="mb-4 border-l-2 border-violet-500/50 pl-4">
                <div className="text-xs text-violet-400 mb-1">You asked</div>
                <div className="text-sm text-stone-300 mb-2">{c.q}</div>
                <div className="text-xs text-stone-500 mb-1">VisionAid</div>
                <div className="text-sm text-stone-200">{c.a}</div>
              </div>
            ))}
          </Modal>
        )}

        {showSettings && (
          <Modal onClose={() => setShowSettings(false)} title="SETTINGS">
            <div className="space-y-6">
              <div>
                <label className="text-xs text-stone-500 uppercase block mb-2">Language</label>
                <div className="flex gap-2">
                  {['en', 'hi'].map(l => (
                    <button key={l} onClick={() => { setLanguage(l); saveProfile({ language: l }); }} className={`px-4 py-2 rounded-lg border text-sm ${language === l ? 'bg-amber-400 border-amber-400 text-black' : 'border-stone-700 text-stone-300'}`}>{l === 'en' ? 'ENGLISH' : 'हिंदी'}</button>
                  ))}
                </div>
              </div>
              <div>
                <label className="text-xs text-stone-500 uppercase block mb-2">Voice Speed — {voiceSpeed.toFixed(1)}x</label>
                <input type="range" min="0.5" max="1.5" step="0.1" value={voiceSpeed} onChange={(e) => { const v = Number(e.target.value); setVoiceSpeed(v); saveProfile({ voiceSpeed: v }); }} className="w-full accent-amber-400" />
              </div>
              <div>
                <label className="text-xs text-stone-500 uppercase block mb-2">Distance Calibration — focal {getFocalPx()}px</label>
                <p className="text-xs text-stone-600 mb-3">Stand exactly this far from the camera, facing it, then tap the distance.</p>
                <div className="grid grid-cols-3 gap-2">{[1, 2, 3].map(m => (<button key={m} disabled={calibrating} onClick={() => calibrateDistance(m)} className="py-3 rounded-lg border border-stone-700 text-xs text-stone-300 hover:border-amber-400 hover:text-amber-300 disabled:opacity-40">{m}m</button>))}</div>
              </div>
              <div>
                <label className="text-xs text-stone-500 uppercase block mb-2">Road-Hazard Model (optional)</label>
                <p className="text-xs text-stone-600">{potholeReady ? "Active." : "Not found. Export a free pothole model from Roboflow to .onnx (opset 12), place at frontend/public/pothole.onnx, reload."}</p>
              </div>
            </div>
          </Modal>
        )}

        {stats && (
          <Modal onClose={() => setStats(null)} title="USAGE DASHBOARD">
            <div className="grid grid-cols-2 gap-4 mb-6">
              <div className="border border-stone-800 rounded-xl p-4 text-center"><div className="text-3xl font-bold text-amber-400">{stats.total_scans}</div><div className="text-xs text-stone-500 mt-1">TOTAL SCANS</div></div>
              <div className="border border-stone-800 rounded-xl p-4 text-center"><div className="text-3xl font-bold text-amber-400">{stats.closest_object_ever_m ?? '—'}m</div><div className="text-xs text-stone-500 mt-1">CLOSEST EVER</div></div>
            </div>
            {(stats.top_classes || []).map(({ cls, count }) => (<div key={cls} className="flex items-center justify-between text-sm text-stone-300 mb-2"><span className="capitalize">{cls}</span><span className="text-amber-400">{count}</span></div>))}
          </Modal>
        )}
      </div>

      {showMap && (
        <WaypointMap
          waypoint={waypoint}
          routePath={routePath}
          userPos={userPos}
          heading={heading}
          navigating={!!routeSteps}
          onSetWaypoint={setWaypointAndClose}
          onStartNav={startNavigation}
          onStopNav={stopNavigation}
          onClose={() => setShowMap(false)}
        />
      )}
    </div>
  );
}

function Modal({ title, onClose, children }) {
  return (
    <div className="fixed inset-0 bg-black/80 flex items-center justify-center p-4 z-50" onClick={onClose}>
      <div className="bg-stone-950 border border-amber-400/40 rounded-2xl w-full max-w-lg max-h-[80vh] flex flex-col" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between px-6 py-4 border-b border-stone-800">
          <h3 className="text-amber-400 text-sm font-bold">{title}</h3>
          <button onClick={onClose} aria-label="Close" className="text-stone-500 hover:text-amber-300"><X className="w-5 h-5" /></button>
        </div>
        <div className="overflow-y-auto p-6">{children}</div>
      </div>
    </div>
  );
}

export default App;