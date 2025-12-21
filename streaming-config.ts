// Video Streaming Configuration - Option B: Backend as Socket.IO Server
// 
// Backend contract:
// - Socket.IO URL: http://<backend-ip>:<port>
// - Event name: "prediction"
// - Payload: { session_id, batch_id, window_start, label, prob_real, prob_fake, confidence, inference_ms, timestamp }
//
// Network requirements:
// - Android: Allow cleartext HTTP via android:usesCleartextTraffic="true"
// - iOS: ATS exception for backend host (or use HTTPS)
// - Phone must reach backend IP on same LAN
//
// Environment-specific URLs:
// - Android Emulator: use 10.0.2.2 instead of localhost
// - iOS Simulator: localhost works fine
// - Physical Device: use your machine's LAN IP (e.g., 192.168.1.x)

export const STREAMING_CONFIG = {
  // HTTP API endpoint for frame ingestion
  BASE_URL: "http://192.168.1.12:8091", // Backend IP:port
  
  // Socket.IO endpoint for real-time predictions (Option B: Backend is server)
  SOCKET_URL: "http://192.168.1.12:8093", // Socket.IO server port
  
  ENDPOINTS: {
    INGEST_FRAME: "/api/video-streaming/ingest/frame",
  },
  
  // Default session ID (mobile generates unique session per user)
  DEFAULT_SESSION_ID: "mobile-session-" + Date.now(),
  
  // Socket.IO options (Option B: websocket-first, client mode)
  SOCKET_OPTIONS: {
    transports: ["websocket"], // Use websocket only (backend is server)
    reconnection: true,
    reconnectionDelay: 1000,
    reconnectionAttempts: 10,
    timeout: 10000,
  },
};

// Helper to get full API URL
export const getStreamingApiUrl = (endpoint: string): string => {
  return `${STREAMING_CONFIG.BASE_URL}${endpoint}`;
};
