package service

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sync"
	"time"

	"video-streaming/internal/config"
	"video-streaming/internal/dto"
	"video-streaming/internal/models"
	"video-streaming/internal/repository"

	"github.com/google/uuid"
)

type SessionService struct {
	sessions   map[string]*models.Session
	mu         sync.RWMutex
	config     *config.Config
	repository *repository.SessionRepository
}

func NewSessionService(cfg *config.Config, repo *repository.SessionRepository) *SessionService {
	return &SessionService{
		sessions:   make(map[string]*models.Session),
		config:     cfg,
		repository: repo,
	}
}

// CreateSession creates a new streaming session
func (s *SessionService) CreateSession(userID string) (*models.Session, error) {
	sessionID := uuid.New().String()
	tmpDir := filepath.Join(s.config.TmpDir, sessionID)

	// Create session directory
	if err := os.MkdirAll(tmpDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create session directory: %w", err)
	}

	now := time.Now().UTC()
	session := &models.Session{
		ID:        sessionID,
		UserID:    userID,
		TmpDir:    tmpDir,
		Status:    models.StatusReady,
		CreatedAt: now,
		UpdatedAt: now,
	}

	// Store in memory
	s.mu.Lock()
	s.sessions[sessionID] = session
	s.mu.Unlock()

	// Persist to database
	sessionDTO := &dto.SessionDTO{
		ID:        session.ID,
		UserID:    session.UserID,
		VideoPath: session.VideoPath,
		Status:    session.Status,
		CreatedAt: session.CreatedAt,
		UpdatedAt: session.UpdatedAt,
	}

	if err := s.repository.CreateSession(context.Background(), sessionDTO); err != nil {
		log.Printf("Failed to persist session to DB: %v", err)
		// Continue anyway, session exists in memory
	}

	log.Printf("Created session: %s for user: %s", sessionID, userID)
	return session, nil
}

// GetSession retrieves a session by ID
func (s *SessionService) GetSession(sessionID string) (*models.Session, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	session, exists := s.sessions[sessionID]
	return session, exists
}

// StartExtraction starts frame extraction for a session
func (s *SessionService) StartExtraction(sessionID, videoPath string) error {
	s.mu.Lock()
	session, exists := s.sessions[sessionID]
	s.mu.Unlock()

	if !exists {
		return fmt.Errorf("session not found")
	}

	// Update video path
	session.VideoPath = videoPath
	session.Status = models.StatusProcessing
	session.UpdatedAt = time.Now().UTC()

	// Update in database
	ctx := context.Background()
	if err := s.repository.UpdateSessionVideoPath(ctx, sessionID, videoPath); err != nil {
		log.Printf("Failed to update video path in DB: %v", err)
	}
	if err := s.repository.UpdateSessionStatus(ctx, sessionID, models.StatusProcessing); err != nil {
		log.Printf("Failed to update session status in DB: %v", err)
	}

	// Create context for this session
	ctx, cancel := context.WithCancel(context.Background())
	session.CancelFunc = cancel

	// Create extractor
	extractor := NewFrameExtractor(videoPath, session.TmpDir, s.config)
	session.Extractor = extractor

	// Create cleanup service
	cleanup := NewFrameCleanup(session.TmpDir, s.config)
	session.Cleanup = cleanup

	// Start extraction in background
	go func() {
		if err := extractor.Start(ctx); err != nil {
			log.Printf("Extraction error for session %s: %v", sessionID, err)
			s.updateSessionStatus(sessionID, models.StatusFailed)
		} else {
			s.updateSessionStatus(sessionID, models.StatusCompleted)
		}
	}()

	// Start cleanup in background
	go cleanup.Start(ctx)

	log.Printf("Started extraction for session: %s", sessionID)
	return nil
}

// DeleteSession deletes a session and cleans up resources
func (s *SessionService) DeleteSession(sessionID string) error {
	s.mu.Lock()
	session, exists := s.sessions[sessionID]
	if exists {
		delete(s.sessions, sessionID)
	}
	s.mu.Unlock()

	if !exists {
		return fmt.Errorf("session not found")
	}

	// Cancel context
	if session.CancelFunc != nil {
		session.CancelFunc()
	}

	// Remove directory
	if err := os.RemoveAll(session.TmpDir); err != nil {
		log.Printf("Warning: failed to remove session directory: %v", err)
	}

	// Delete from database
	if err := s.repository.DeleteSession(context.Background(), sessionID); err != nil {
		log.Printf("Failed to delete session from DB: %v", err)
	}

	log.Printf("Deleted session: %s", sessionID)
	return nil
}

// Shutdown gracefully shuts down all sessions
func (s *SessionService) Shutdown() {
	s.mu.Lock()
	defer s.mu.Unlock()

	for _, session := range s.sessions {
		if session.CancelFunc != nil {
			session.CancelFunc()
		}
		os.RemoveAll(session.TmpDir)
	}

	log.Println("All sessions cleaned up")
}

func (s *SessionService) updateSessionStatus(sessionID, status string) {
	s.mu.Lock()
	if session, exists := s.sessions[sessionID]; exists {
		session.Status = status
		session.UpdatedAt = time.Now().UTC()
	}
	s.mu.Unlock()

	// Update in database
	if err := s.repository.UpdateSessionStatus(context.Background(), sessionID, status); err != nil {
		log.Printf("Failed to update session status in DB: %v", err)
	}
}
