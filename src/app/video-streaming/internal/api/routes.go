package api

import (
	"net/http"
	"strings"
)

func SetupRoutes(sessionHandler *Handler) http.Handler {
	mux := http.NewServeMux()

	// Health check
	mux.HandleFunc("/api/video-streaming/health", sessionHandler.HealthCheck)

	// Aggregator endpoints
	mux.HandleFunc("/api/video-streaming/ingest/frame", sessionHandler.IngestFrame)
	mux.HandleFunc("/api/video-streaming/stats/aggregator", sessionHandler.GetAggregatorStats)

	// WebRTC streaming endpoints (NEW: low-latency alternative to IngestFrame)
	mux.HandleFunc("/api/video-streaming/webrtc/stream/offer", sessionHandler.StartWebRTCStream)
	mux.HandleFunc("/api/video-streaming/webrtc/stream/candidate", sessionHandler.HandleICECandidate)
	mux.HandleFunc("/api/video-streaming/webrtc/stream/", func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/close") {
			sessionHandler.CloseWebRTCStream(w, r)
		} else {
			http.NotFound(w, r)
		}
	})
	mux.HandleFunc("/api/video-streaming/webrtc/stats", sessionHandler.GetWebRTCStats)

	// Serve stored frames (Option 2: static file serving)
	mux.HandleFunc("/frames/", sessionHandler.ServeFrame)

	// Frame upload endpoints (legacy)
	mux.HandleFunc("/api/video-streaming/sessions/", func(w http.ResponseWriter, r *http.Request) {
		// Route to appropriate handler based on path
		if strings.Contains(r.URL.Path, "/frames/batch") {
			sessionHandler.UploadFrameBatch(w, r)
		} else if strings.Contains(r.URL.Path, "/frames") {
			sessionHandler.UploadFrame(w, r)
		} else {
			http.NotFound(w, r)
		}
	})

	// Apply middleware
	handler := LoggingMiddleware(mux)
	handler = RecoveryMiddleware(handler)
	handler = CORSMiddleware(handler)

	return handler
}
