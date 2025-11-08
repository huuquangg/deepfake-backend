package api

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"video-streaming/internal/dto"
	"video-streaming/internal/service"
)

type Handler struct {
	sessionService *service.SessionService
}

func NewHandler(sessionService *service.SessionService) *Handler {
	return &Handler{
		sessionService: sessionService,
	}
}

func (h *Handler) HealthCheck(w http.ResponseWriter, r *http.Request) {
	response := dto.HealthResponse{
		Status:    "healthy",
		Timestamp: time.Now().UTC().Format(time.RFC3339),
		Version:   "1.0.0",
	}
	h.respondJSON(w, http.StatusOK, response)
}

func (h *Handler) HandleSessions(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodPost:
		h.CreateSession(w, r)
	default:
		h.respondError(w, http.StatusMethodNotAllowed, "Method not allowed")
	}
}

func (h *Handler) HandleSession(w http.ResponseWriter, r *http.Request) {
	sessionID := h.extractSessionID(r.URL.Path)
	if sessionID == "" {
		h.respondError(w, http.StatusBadRequest, "Session ID required")
		return
	}

	switch r.Method {
	case http.MethodPost:
		h.UploadVideo(w, r, sessionID)
	case http.MethodGet:
		h.GetSession(w, r, sessionID)
	case http.MethodDelete:
		h.DeleteSession(w, r, sessionID)
	default:
		h.respondError(w, http.StatusMethodNotAllowed, "Method not allowed")
	}
}

func (h *Handler) CreateSession(w http.ResponseWriter, r *http.Request) {
	var req dto.CreateSessionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		h.respondError(w, http.StatusBadRequest, "Invalid request body")
		return
	}

	if req.UserID == "" {
		h.respondError(w, http.StatusBadRequest, "user_id is required")
		return
	}

	session, err := h.sessionService.CreateSession(req.UserID)
	if err != nil {
		h.respondError(w, http.StatusInternalServerError, err.Error())
		return
	}

	response := dto.CreateSessionResponse{
		SessionID: session.ID,
		Status:    session.Status,
		CreatedAt: session.CreatedAt.Format(time.RFC3339),
	}

	h.respondJSON(w, http.StatusCreated, response)
}

func (h *Handler) UploadVideo(w http.ResponseWriter, r *http.Request, sessionID string) {
	session, exists := h.sessionService.GetSession(sessionID)
	if !exists {
		h.respondError(w, http.StatusNotFound, "Session not found")
		return
	}

	// Save uploaded video
	videoPath := filepath.Join(session.TmpDir, "input_video.mp4")
	videoFile, err := os.Create(videoPath)
	if err != nil {
		h.respondError(w, http.StatusInternalServerError, fmt.Sprintf("Failed to create video file: %v", err))
		return
	}
	defer videoFile.Close()

	// Copy video data
	_, err = io.Copy(videoFile, r.Body)
	if err != nil {
		h.respondError(w, http.StatusInternalServerError, fmt.Sprintf("Failed to save video: %v", err))
		return
	}

	log.Printf("Video uploaded for session %s", sessionID)

	// Start extraction
	if err := h.sessionService.StartExtraction(sessionID, videoPath); err != nil {
		h.respondError(w, http.StatusInternalServerError, err.Error())
		return
	}

	response := dto.UploadVideoResponse{
		Message:   "Video processing started",
		SessionID: sessionID,
		Status:    "processing",
	}

	h.respondJSON(w, http.StatusOK, response)
}

func (h *Handler) GetSession(w http.ResponseWriter, r *http.Request, sessionID string) {
	session, exists := h.sessionService.GetSession(sessionID)
	if !exists {
		h.respondError(w, http.StatusNotFound, "Session not found")
		return
	}

	response := dto.SessionDTO{
		ID:        session.ID,
		UserID:    session.UserID,
		VideoPath: session.VideoPath,
		Status:    session.Status,
		CreatedAt: session.CreatedAt,
		UpdatedAt: session.UpdatedAt,
	}

	h.respondJSON(w, http.StatusOK, response)
}

func (h *Handler) DeleteSession(w http.ResponseWriter, r *http.Request, sessionID string) {
	if err := h.sessionService.DeleteSession(sessionID); err != nil {
		h.respondError(w, http.StatusNotFound, err.Error())
		return
	}

	response := dto.SuccessResponse{
		Message: "Session deleted successfully",
	}

	h.respondJSON(w, http.StatusOK, response)
}

func (h *Handler) extractSessionID(path string) string {
	parts := strings.Split(strings.Trim(path, "/"), "/")
	if len(parts) >= 4 {
		return parts[3]
	}
	return ""
}

func (h *Handler) respondJSON(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}

func (h *Handler) respondError(w http.ResponseWriter, status int, message string) {
	h.respondJSON(w, status, dto.ErrorResponse{
		Error:   http.StatusText(status),
		Message: message,
		Code:    status,
	})
}
