package api

import (
	"net/http"
)

func SetupRoutes(sessionHandler *Handler) http.Handler {
	mux := http.NewServeMux()

	// Health check
	mux.HandleFunc("/health", sessionHandler.HealthCheck)

	// Session endpoints
	// mux.HandleFunc("/api/v1/sessions", h.HandleSessions)
	// mux.HandleFunc("/api/v1/sessions/", h.HandleSession)

	// Apply middleware
	handler := LoggingMiddleware(mux)
	handler = RecoveryMiddleware(handler)

	return handler
}
