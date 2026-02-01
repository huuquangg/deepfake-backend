package models

import (
	"context"
	"time"
)

// Session represents an active streaming session
type Session struct {
	ID         string
	UserID     string
	TmpDir     string
	VideoPath  string
	Status     string
	CreatedAt  time.Time
	UpdatedAt  time.Time
	Extractor  FrameExtractor
	Cleanup    FrameCleaner
	CancelFunc context.CancelFunc
}

// FrameExtractor interface for frame extraction service
type FrameExtractor interface {
	Start(ctx context.Context) error
}

// FrameCleaner interface for cleanup service
type FrameCleaner interface {
	Start(ctx context.Context)
}

// Session statuses
const (
	StatusReady      = "ready"
	StatusProcessing = "processing"
	StatusCompleted  = "completed"
	StatusFailed     = "failed"
)