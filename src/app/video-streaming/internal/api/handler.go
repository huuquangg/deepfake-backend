package api

import (
	"encoding/json"
	"net/http"
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

func (handler *Handler) HealthCheck(w http.ResponseWriter, r *http.Request) {
	response := dto.HealthResponse{
		Status:    "healthy",
		Timestamp: time.Now().UTC().Format(time.RFC3339),
		Version:   "1.0.0",
	}
	handler.respondJSON(w, http.StatusOK, response)
}

// func (handler *Handler) HandleSession(w http.ResponseWriter, r *http.Request) {
// 	sessionID := handler.extractSessionID(r.URL.Path)
// 	if sessionID == "" {
// 		handler.respondError(w, http.StatusBadRequest, "Session ID required")
// 		return
// 	}

// 	switch r.Method {
// 	case http.MethodPost:
// 		handler.UploadVideo(w, r, sessionID)
// 	case http.MethodGet:
// 		handler.GetSession(w, r, sessionID)
// 	case http.MethodDelete:
// 		handler.DeleteSession(w, r, sessionID)
// 	default:
// 		handler.respondError(w, http.StatusMethodNotAllowed, "Method not allowed")
// 	}
// }

// func (handler *Handler) CreateSession(w http.ResponseWriter, r *http.Request) {
// 	var req dto.CreateSessionRequest
// 	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
// 		handler.respondError(w, http.StatusBadRequest, "Invalid request body")
// 		return
// 	}

// 	if req.UserID == "" {
// 		handler.respondError(w, http.StatusBadRequest, "user_id is required")
// 		return
// 	}

// 	session, err := handler.sessionService.CreateSession(req.UserID)
// 	if err != nil {
// 		handler.respondError(w, http.StatusInternalServerError, err.Error())
// 		return
// 	}

// 	response := dto.CreateSessionResponse{
// 		SessionID: session.ID,
// 		Status:    session.Status,
// 		CreatedAt: session.CreatedAt.Format(time.RFC3339),
// 	}

// 	handler.respondJSON(w, http.StatusCreated, response)
// }

// func (handler *Handler) UploadVideo(w http.ResponseWriter, r *http.Request, sessionID string) {
// 	session, exists := handler.sessionService.GetSession(sessionID)
// 	if !exists {
// 		handler.respondError(w, http.StatusNotFound, "Session not found")
// 		return
// 	}

// 	// Save uploaded video
// 	videoPath := filepathandler.Join(session.TmpDir, "input_video.mp4")
// 	videoFile, err := os.Create(videoPath)
// 	if err != nil {
// 		handler.respondError(w, http.StatusInternalServerError, fmt.Sprintf("Failed to create video file: %v", err))
// 		return
// 	}
// 	defer videoFile.Close()

// 	// Copy video data
// 	_, err = io.Copy(videoFile, r.Body)
// 	if err != nil {
// 		handler.respondError(w, http.StatusInternalServerError, fmt.Sprintf("Failed to save video: %v", err))
// 		return
// 	}

// 	log.Printf("Video uploaded for session %s", sessionID)

// 	// Start extraction
// 	if err := handler.sessionService.StartExtraction(sessionID, videoPath); err != nil {
// 		handler.respondError(w, http.StatusInternalServerError, err.Error())
// 		return
// 	}

// 	response := dto.UploadVideoResponse{
// 		Message:   "Video processing started",
// 		SessionID: sessionID,
// 		Status:    "processing",
// 	}

// 	handler.respondJSON(w, http.StatusOK, response)
// }

// func (handler *Handler) GetSession(w http.ResponseWriter, r *http.Request, sessionID string) {
// 	session, exists := handler.sessionService.GetSession(sessionID)
// 	if !exists {
// 		handler.respondError(w, http.StatusNotFound, "Session not found")
// 		return
// 	}

// 	response := dto.SessionDTO{
// 		ID:        session.ID,
// 		UserID:    session.UserID,
// 		VideoPath: session.VideoPath,
// 		Status:    session.Status,
// 		CreatedAt: session.CreatedAt,
// 		UpdatedAt: session.UpdatedAt,
// 	}

// 	handler.respondJSON(w, http.StatusOK, response)
// }

// func (handler *Handler) DeleteSession(w http.ResponseWriter, r *http.Request, sessionID string) {
// 	if err := handler.sessionService.DeleteSession(sessionID); err != nil {
// 		handler.respondError(w, http.StatusNotFound, err.Error())
// 		return
// 	}

// 	response := dto.SuccessResponse{
// 		Message: "Session deleted successfully",
// 	}

// 	handler.respondJSON(w, http.StatusOK, response)
// }

// func (handler *Handler) extractSessionID(path string) string {
// 	parts := strings.Split(strings.Trim(path, "/"), "/")
// 	if len(parts) >= 4 {
// 		return parts[3]
// 	}
// 	return ""
// }

func (handler *Handler) respondJSON(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}

func (handler *Handler) respondError(w http.ResponseWriter, status int, message string) {
	handler.respondJSON(w, status, dto.ErrorResponse{
		Error:   http.StatusText(status),
		Message: message,
		Code:    status,
	})
}
