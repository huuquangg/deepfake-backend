package api

import (
	"net/http"
	"time"

	"api-gateway/internal/config"
)

type Route struct {
	Prefix      string
	Target      string
	StripPrefix string
}

func Routes(cfg *config.Config) []Route {
	return []Route{
		// Video Streaming Service
		{Prefix: "/api/video-streaming/", Target: cfg.VideoStreamingURL.String(), StripPrefix: ""},
		{Prefix: "/frames/", Target: cfg.VideoStreamingURL.String(), StripPrefix: ""},
		
		// Core Banking Service
		{Prefix: "/api/core-banking/", Target: cfg.CoreBankingURL.String(), StripPrefix: ""},
		
		// OpenFace Feature Extraction (API endpoints)
		{Prefix: "/api/openface/", Target: cfg.OpenFaceURL.String(), StripPrefix: "/api/openface"},
		
		// OpenFace Batch API
		{Prefix: "/api/openface-batch/", Target: cfg.OpenFaceBatchURL.String(), StripPrefix: "/api/openface-batch"},
		
		// Frequency Feature Extraction
		{Prefix: "/api/frequency/", Target: cfg.FrequencyURL.String(), StripPrefix: "/api/frequency"},
		
		// Socket.IO for real-time predictions
		{Prefix: "/socket.io/", Target: cfg.VideoStreamingSocket.String(), StripPrefix: ""},
	}
}

func SetupRoutes(routes []Route) http.Handler {
	mux := http.NewServeMux()

	for _, route := range routes {
		proxy := newReverseProxy(route.Target, route.StripPrefix)
		mux.Handle(route.Prefix, proxy)
	}

	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		response := map[string]string{
			"status": "ok",
			"time":   time.Now().UTC().Format(time.RFC3339),
		}
		writeJSON(w, http.StatusOK, response)
	})

	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		response := map[string]any{
			"service": "api-gateway",
			"routes":  RoutePrefixes(routes),
		}
		writeJSON(w, http.StatusOK, response)
	})

	return mux
}

func RoutePrefixes(routes []Route) []string {
	prefixes := make([]string, 0, len(routes))
	for _, route := range routes {
		prefixes = append(prefixes, route.Prefix)
	}
	return prefixes
}
