package api

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/pion/webrtc/v3"
	"video-streaming/internal/aggregator"
	"video-streaming/internal/config"
	"video-streaming/internal/dto"
	"video-streaming/internal/service"
	webrtcHandler "video-streaming/internal/webrtc"
)

type Handler struct {
	sessionService *service.SessionService
	frameCounters  sync.Map // map[sessionID]int64 - atomic counter per session
	aggregator     *aggregator.Aggregator
	config         *config.Config
	webrtcHandler  *webrtcHandler.StreamHandler // WebRTC stream handler
}

// Constructor for Handler
func NewHandler(sessionService *service.SessionService, agg *aggregator.Aggregator, cfg *config.Config) *Handler {
	return &Handler{
		sessionService: sessionService,
		aggregator:     agg,
		config:         cfg,
		webrtcHandler:  webrtcHandler.NewStreamHandler(agg),
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

// UploadFrame godoc
// @Summary      Upload a single frame image
// @Description  Upload a frame image for a session with timestamp. Optimized for high throughput (20-30 fps)
// @Tags         Frames
// @Accept       multipart/form-data
// @Produce      json
// @Param        session_id  path      string  true  "Session ID"
// @Param        frame       formData  file    true  "Frame image file"
// @Success      200         {object}  dto.UploadFrameResponse
// @Failure      400         {object}  dto.ErrorResponse
// @Failure      500         {object}  dto.ErrorResponse
// @Router       /api/v1/sessions/{session_id}/frames [post]
func (handler *Handler) UploadFrame(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		handler.respondError(w, http.StatusMethodNotAllowed, "Method not allowed")
		return
	}

	// Extract session ID from URL path
	sessionID := handler.extractSessionIDFromPath(r.URL.Path)
	if sessionID == "" {
		handler.respondError(w, http.StatusBadRequest, "Session ID is required")
		return
	}

	// Parse multipart form with 32MB max memory
	if err := r.ParseMultipartForm(32 << 20); err != nil {
		handler.respondError(w, http.StatusBadRequest, fmt.Sprintf("Failed to parse form: %v", err))
		return
	}

	// Get the frame file
	file, header, err := r.FormFile("frame")
	if err != nil {
		handler.respondError(w, http.StatusBadRequest, fmt.Sprintf("Failed to get frame file: %v", err))
		return
	}
	defer file.Close()

	// Validate file type
	if !isValidImageType(header.Filename) {
		handler.respondError(w, http.StatusBadRequest, "Invalid file type. Only JPEG/JPG/PNG images are allowed")
		return
	}

	// Create storage directory if with sessionId
	storageDir := fmt.Sprintf("[Storage]/%s", sessionID)
	if err := os.MkdirAll(storageDir, 0755); err != nil {
		handler.respondError(w, http.StatusInternalServerError, fmt.Sprintf("Failed to create storage directory: %v", err))
		return
	}

	// Generate filename with pattern: {SessionID}_{timestamp}_{counter}.jpeg
	// Counter ensures no overwrites and maintains order
	filename := handler.generateFrameFilename(sessionID)
	filePath := filepath.Join(storageDir, filename)

	// Create the file
	destFile, err := os.Create(filePath)
	if err != nil {
		handler.respondError(w, http.StatusInternalServerError, fmt.Sprintf("Failed to create file: %v", err))
		return
	}
	defer destFile.Close()

	// Copy the uploaded file to destination (optimized buffered copy)
	if _, err := io.Copy(destFile, file); err != nil {
		handler.respondError(w, http.StatusInternalServerError, fmt.Sprintf("Failed to save file: %v", err))
		return
	}

	log.Printf("Frame uploaded: %s for session %s", filename, sessionID)

	// Respond with success
	response := dto.UploadFrameResponse{
		Message:   "Frame uploaded successfully",
		SessionID: sessionID,
		FramePath: filePath,
		Timestamp: time.Now().UTC().Format(time.RFC3339),
	}

	handler.respondJSON(w, http.StatusOK, response)
}

// UploadFrameBatch godoc
// @Summary      Upload multiple frame images
// @Description  Upload multiple frame images in a single request for better throughput
// @Tags         Frames
// @Accept       multipart/form-data
// @Produce      json
// @Param        session_id  path      string  true  "Session ID"
// @Param        frames      formData  file    true  "Frame image files" multiple
// @Success      200         {object}  dto.FrameUploadBatchResponse
// @Failure      400         {object}  dto.ErrorResponse
// @Failure      500         {object}  dto.ErrorResponse
// @Router       /api/v1/sessions/{session_id}/frames/batch [post]
func (handler *Handler) UploadFrameBatch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		handler.respondError(w, http.StatusMethodNotAllowed, "Method not allowed")
		return
	}

	// Extract session ID from URL path
	sessionID := handler.extractSessionIDFromPath(r.URL.Path)
	if sessionID == "" {
		handler.respondError(w, http.StatusBadRequest, "Session ID is required")
		return
	}

	// Parse multipart form with 128MB max memory for batch upload
	if err := r.ParseMultipartForm(128 << 20); err != nil {
		handler.respondError(w, http.StatusBadRequest, fmt.Sprintf("Failed to parse form: %v", err))
		return
	}

	// Create storage directory if not exists
	storageDir := fmt.Sprintf("[Storage]/%s", sessionID)
	if err := os.MkdirAll(storageDir, 0755); err != nil {
		handler.respondError(w, http.StatusInternalServerError, fmt.Sprintf("Failed to create storage directory: %v", err))
		return
	}

	// Get all uploaded files
	files := r.MultipartForm.File["frames"]
	if len(files) == 0 {
		handler.respondError(w, http.StatusBadRequest, "No frames provided")
		return
	}

	// Process files concurrently for better performance
	var wg sync.WaitGroup
	var mu sync.Mutex
	framePaths := make([]string, 0, len(files))
	errorChan := make(chan error, len(files))

	for _, fileHeader := range files {
		wg.Add(1)
		go func(fh *multipart.FileHeader) {
			defer wg.Done()

			// Validate file type
			if !isValidImageType(fh.Filename) {
				errorChan <- fmt.Errorf("invalid file type: %s", fh.Filename)
				return
			}

			// Open the file
			file, err := fh.Open()
			if err != nil {
				errorChan <- fmt.Errorf("failed to open file %s: %v", fh.Filename, err)
				return
			}
			defer file.Close()

			// Generate filename with counter to prevent overwrites
			filename := handler.generateFrameFilename(sessionID)
			filePath := filepath.Join(storageDir, filename)

			// Create destination file
			destFile, err := os.Create(filePath)
			if err != nil {
				errorChan <- fmt.Errorf("failed to create file %s: %v", filename, err)
				return
			}
			defer destFile.Close()

			// Copy file
			if _, err := io.Copy(destFile, file); err != nil {
				errorChan <- fmt.Errorf("failed to save file %s: %v", filename, err)
				return
			}

			// Add to results
			mu.Lock()
			framePaths = append(framePaths, filePath)
			mu.Unlock()
		}(fileHeader)
	}

	// Wait for all goroutines to complete
	wg.Wait()
	close(errorChan)

	// Check for errors
	var errors []string
	for err := range errorChan {
		errors = append(errors, err.Error())
	}

	if len(errors) > 0 {
		log.Printf("Batch upload errors: %v", errors)
		// Return partial success if some files were uploaded
		if len(framePaths) > 0 {
			log.Printf("Partial success: %d/%d frames uploaded", len(framePaths), len(files))
		}
	}

	if len(framePaths) == 0 {
		handler.respondError(w, http.StatusInternalServerError, "Failed to upload any frames")
		return
	}

	log.Printf("Batch upload complete: %d frames for session %s", len(framePaths), sessionID)

	// Respond with success
	response := dto.FrameUploadBatchResponse{
		Message:       fmt.Sprintf("Uploaded %d frames successfully", len(framePaths)),
		SessionID:     sessionID,
		TotalUploaded: len(framePaths),
		FramePaths:    framePaths,
	}

	handler.respondJSON(w, http.StatusOK, response)
}

// Helper methods for responses
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

// extractSessionIDFromPath extracts session ID from URL path
func (handler *Handler) extractSessionIDFromPath(path string) string {
	// Expected patterns:
	// /api/v1/sessions/{session_id}/frames
	// /api/v1/sessions/{session_id}/frames/batch
	parts := strings.Split(strings.Trim(path, "/"), "/")
	for i, part := range parts {
		if part == "sessions" && i+1 < len(parts) {
			return parts[i+1]
		}
	}
	return ""
}

// isValidImageType checks if the file has a valid image extension
func isValidImageType(filename string) bool {
	ext := strings.ToLower(filepath.Ext(filename))
	return ext == ".jpg" || ext == ".jpeg" || ext == ".png"
}

// generateFrameFilename generates a unique ordered filename for a frame
// Pattern: {SessionID}_{timestamp}_{counter}.jpeg
// This prevents overwrites and maintains insertion order
func (handler *Handler) generateFrameFilename(sessionID string) string {
	// Get or initialize counter for this session
	var counter int64
	if val, ok := handler.frameCounters.Load(sessionID); ok {
		counter = val.(int64)
	}

	// Increment counter atomically
	counter++
	handler.frameCounters.Store(sessionID, counter)

	// Generate filename with timestamp and counter
	timestamp := time.Now().UnixNano() / int64(time.Millisecond)
	filename := fmt.Sprintf("%s_%d_%06d.jpeg", sessionID, timestamp, counter)

	return filename
}

// IngestFrame godoc
// @Summary      Ingest a single frame for aggregation
// @Description  Non-blocking endpoint that accepts a frame and adds it to the aggregation buffer
// @Tags         Frames
// @Accept       multipart/form-data
// @Produce      json
// @Param        session_id  formData  string  true   "Session ID"
// @Param        frame       formData  file    true   "Frame image file"
// @Success      200         {object}  dto.UploadFrameResponse
// @Failure      400         {object}  dto.ErrorResponse
// @Failure      500         {object}  dto.ErrorResponse
// @Router       /api/video-streaming/ingest/frame [post]
func (handler *Handler) IngestFrame(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		handler.respondError(w, http.StatusMethodNotAllowed, "Method not allowed")
		return
	}

	// Parse multipart form with 32MB max memory
	if err := r.ParseMultipartForm(32 << 20); err != nil {
		handler.respondError(w, http.StatusBadRequest, fmt.Sprintf("Failed to parse form: %v", err))
		return
	}

	// Get session_id from form
	sessionID := r.FormValue("session_id")
	if sessionID == "" {
		handler.respondError(w, http.StatusBadRequest, "session_id is required")
		return
	}

	// Get the frame file
	file, header, err := r.FormFile("frame")
	if err != nil {
		handler.respondError(w, http.StatusBadRequest, fmt.Sprintf("Failed to get frame file: %v", err))
		return
	}
	defer file.Close()

	// Validate file type
	if !isValidImageType(header.Filename) {
		handler.respondError(w, http.StatusBadRequest, "Invalid file type. Only JPEG/JPG/PNG images are allowed")
		return
	}

	// Read file into memory
	frameData, err := io.ReadAll(file)
	if err != nil {
		handler.respondError(w, http.StatusInternalServerError, fmt.Sprintf("Failed to read frame: %v", err))
		return
	}

	// Generate unique filename
	filename := handler.generateFrameFilename(sessionID)

	// Ingest frame into aggregator (non-blocking)
	if err := handler.aggregator.IngestFrame(sessionID, frameData, filename); err != nil {
		handler.respondError(w, http.StatusInternalServerError, fmt.Sprintf("Failed to ingest frame: %v", err))
		return
	}

	// Return immediately (non-blocking)
	response := dto.UploadFrameResponse{
		Message:   "Frame ingested successfully",
		SessionID: sessionID,
		FramePath: filename,
		Timestamp: time.Now().UTC().Format(time.RFC3339),
	}

	handler.respondJSON(w, http.StatusOK, response)
}

// GetAggregatorStats godoc
// @Summary      Get aggregator statistics
// @Description  Returns current aggregator state including active sessions and buffered frames
// @Tags         Stats
// @Produce      json
// @Success      200  {object}  map[string]interface{}
// @Router       /api/video-streaming/stats/aggregator [get]
func (handler *Handler) GetAggregatorStats(w http.ResponseWriter, r *http.Request) {
	stats := handler.aggregator.GetSessionStats()
	handler.respondJSON(w, http.StatusOK, stats)
}

// ServeFrame godoc
// @Summary      Serve stored frame image
// @Description  Serves frame images from disk storage (Option 2 frame pointers)
// @Tags         Frames
// @Produce      image/jpeg
// @Param        session_id  path      string  true  "Session ID"
// @Param        batch_id    path      string  true  "Batch ID"
// @Param        filename    path      string  true  "Frame filename"
// @Success      200  {file}  binary
// @Failure      404  {object}  dto.ErrorResponse
// @Failure      500  {object}  dto.ErrorResponse
// @Router       /frames/{session_id}/{batch_id}/{filename} [get]
func (handler *Handler) ServeFrame(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		handler.respondError(w, http.StatusMethodNotAllowed, "Method not allowed")
		return
	}

	// Parse path: /frames/<session_id>/<batch_id>/<filename>
	pathParts := strings.Split(strings.TrimPrefix(r.URL.Path, "/frames/"), "/")
	if len(pathParts) != 3 {
		handler.respondError(w, http.StatusBadRequest, "Invalid frame path format. Expected /frames/<session_id>/<batch_id>/<filename>")
		return
	}

	sessionID := pathParts[0]
	batchID := pathParts[1]
	filename := pathParts[2]

	// Sanitize inputs to prevent path traversal
	if strings.Contains(sessionID, "..") || strings.Contains(batchID, "..") || strings.Contains(filename, "..") {
		handler.respondError(w, http.StatusBadRequest, "Invalid path: contains path traversal")
		return
	}

	// Build file path from storage config
	storageDir := handler.config.FrameStorageDir
	framePath := filepath.Join(storageDir, sessionID, batchID, filename)

	// Check if file exists
	fileInfo, err := os.Stat(framePath)
	if os.IsNotExist(err) {
		handler.respondError(w, http.StatusNotFound, fmt.Sprintf("Frame not found: %s", filename))
		return
	}
	if err != nil {
		handler.respondError(w, http.StatusInternalServerError, fmt.Sprintf("Error accessing frame: %v", err))
		return
	}

	// Ensure it's a file, not directory
	if fileInfo.IsDir() {
		handler.respondError(w, http.StatusBadRequest, "Path is a directory, not a file")
		return
	}

	// Serve the file
	w.Header().Set("Content-Type", "image/jpeg")
	w.Header().Set("Cache-Control", "public, max-age=3600")  // Cache for 1 hour
	http.ServeFile(w, r, framePath)
}

// WebRTC Streaming Endpoints

// StartWebRTCStream godoc
// @Summary      Start WebRTC video stream
// @Description  Establish WebRTC connection for real-time video streaming with lower latency than HTTP
// @Tags         WebRTC
// @Accept       json
// @Produce      json
// @Param        offer  body      object  true  "WebRTC offer with session_id and SDP"
// @Success      200    {object}  object  "WebRTC answer with SDP"
// @Failure      400    {object}  dto.ErrorResponse
// @Failure      500    {object}  dto.ErrorResponse
// @Router       /api/video-streaming/webrtc/stream/offer [post]
func (handler *Handler) StartWebRTCStream(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		handler.respondError(w, http.StatusMethodNotAllowed, "Method not allowed")
		return
	}

	// Parse request body
	var offer struct {
		SessionID string `json:"session_id"`
		SDP       string `json:"sdp"`
		Type      string `json:"type"`
	}

	if err := json.NewDecoder(r.Body).Decode(&offer); err != nil {
		handler.respondError(w, http.StatusBadRequest, fmt.Sprintf("Failed to parse offer: %v", err))
		return
	}

	if offer.SessionID == "" {
		handler.respondError(w, http.StatusBadRequest, "session_id is required")
		return
	}

	if offer.SDP == "" {
		handler.respondError(w, http.StatusBadRequest, "sdp is required")
		return
	}

	log.Printf("Received WebRTC offer from session: %s", offer.SessionID)

	// Handle the offer and create answer
	answerSDP, err := handler.webrtcHandler.HandleOffer(offer.SessionID, offer.SDP)
	if err != nil {
		handler.respondError(w, http.StatusInternalServerError, fmt.Sprintf("Failed to handle offer: %v", err))
		return
	}

	// Return answer
	answer := map[string]interface{}{
		"session_id": offer.SessionID,
		"sdp":        answerSDP,
		"type":       "answer",
	}

	log.Printf("Sending WebRTC answer to session: %s", offer.SessionID)
	handler.respondJSON(w, http.StatusOK, answer)
}

// HandleICECandidate godoc
// @Summary      Add ICE candidate for WebRTC connection
// @Description  Receive and add ICE candidates during WebRTC negotiation
// @Tags         WebRTC
// @Accept       json
// @Produce      json
// @Param        candidate  body      object  true  "ICE candidate with session_id"
// @Success      200        {object}  object  "Success message"
// @Failure      400        {object}  dto.ErrorResponse
// @Failure      500        {object}  dto.ErrorResponse
// @Router       /api/video-streaming/webrtc/stream/candidate [post]
func (handler *Handler) HandleICECandidate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		handler.respondError(w, http.StatusMethodNotAllowed, "Method not allowed")
		return
	}

	// Parse request body
	var req struct {
		SessionID string                  `json:"session_id"`
		Candidate webrtc.ICECandidateInit `json:"candidate"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		handler.respondError(w, http.StatusBadRequest, fmt.Sprintf("Failed to parse candidate: %v", err))
		return
	}

	if req.SessionID == "" {
		handler.respondError(w, http.StatusBadRequest, "session_id is required")
		return
	}

	// Add ICE candidate
	if err := handler.webrtcHandler.HandleICECandidate(req.SessionID, req.Candidate); err != nil {
		handler.respondError(w, http.StatusInternalServerError, fmt.Sprintf("Failed to add ICE candidate: %v", err))
		return
	}

	handler.respondJSON(w, http.StatusOK, map[string]interface{}{
		"message":    "ICE candidate added successfully",
		"session_id": req.SessionID,
	})
}

// CloseWebRTCStream godoc
// @Summary      Close WebRTC stream
// @Description  Close the WebRTC connection for a session
// @Tags         WebRTC
// @Produce      json
// @Param        session_id  path      string  true  "Session ID"
// @Success      200         {object}  object  "Success message"
// @Failure      400         {object}  dto.ErrorResponse
// @Failure      500         {object}  dto.ErrorResponse
// @Router       /api/video-streaming/webrtc/stream/{session_id}/close [post]
func (handler *Handler) CloseWebRTCStream(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		handler.respondError(w, http.StatusMethodNotAllowed, "Method not allowed")
		return
	}

	// Extract session ID from URL path
	sessionID := handler.extractSessionIDFromWebRTCPath(r.URL.Path)
	if sessionID == "" {
		handler.respondError(w, http.StatusBadRequest, "session_id is required")
		return
	}

	// Close WebRTC session
	if err := handler.webrtcHandler.CloseSession(sessionID); err != nil {
		handler.respondError(w, http.StatusInternalServerError, fmt.Sprintf("Failed to close session: %v", err))
		return
	}

	log.Printf("WebRTC session closed: %s", sessionID)
	handler.respondJSON(w, http.StatusOK, map[string]interface{}{
		"message":    "WebRTC session closed successfully",
		"session_id": sessionID,
	})
}

// GetWebRTCStats godoc
// @Summary      Get WebRTC statistics
// @Description  Returns statistics for all active WebRTC sessions
// @Tags         WebRTC
// @Produce      json
// @Success      200  {object}  map[string]interface{}
// @Router       /api/video-streaming/webrtc/stats [get]
func (handler *Handler) GetWebRTCStats(w http.ResponseWriter, r *http.Request) {
	stats := handler.webrtcHandler.GetSessionStats()
	handler.respondJSON(w, http.StatusOK, stats)
}

// extractSessionIDFromWebRTCPath extracts session ID from WebRTC path
func (handler *Handler) extractSessionIDFromWebRTCPath(path string) string {
	// Expected pattern: /api/video-streaming/webrtc/stream/{session_id}/close
	parts := strings.Split(strings.Trim(path, "/"), "/")
	for i, part := range parts {
		if part == "stream" && i+1 < len(parts) {
			return parts[i+1]
		}
	}
	return ""
}
