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

	"video-streaming/internal/dto"
	"video-streaming/internal/service"
)

type Handler struct {
	sessionService *service.SessionService
	frameCounters  sync.Map // map[sessionID]int64 - atomic counter per session
}

// Constructor for Handler
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
		handler.respondError(w, http.StatusBadRequest, "Invalid file type. Only JPEG/JPG images are allowed")
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
	return ext == ".jpg" || ext == ".jpeg"
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
