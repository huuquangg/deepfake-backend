package service

import (
	"os"
	"sync"

	"video-streaming/internal/config"
	"video-streaming/internal/models"
	"video-streaming/internal/repository"
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

func (s *SessionService) Shutdown() {
	s.mu.Lock()
	defer s.mu.Unlock()

	for _, session := range s.sessions {
		if session.CancelFunc != nil {
			session.CancelFunc()
		}
		os.RemoveAll(session.TmpDir)
	}
}
