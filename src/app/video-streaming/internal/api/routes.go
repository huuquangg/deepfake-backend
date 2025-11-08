package api

import (
	"net/http"
)

func SetupRoutes(h *Handler) http.Handler {
	mux := http.NewServeMux()

	// Health check
	mux.HandleFunc("/health", h.HealthCheck)

	// Session endpoints
	mux.HandleFunc("/api/v1/sessions", h.HandleSessions)
	mux.HandleFunc("/api/v1/sessions/", h.HandleSession)

	// Apply middleware
	handler := LoggingMiddleware(mux)
	handler = RecoveryMiddleware(handler)

	return handler
}
