package service

import (
	"context"
	"log"
	"os"
	"path/filepath"
	"time"

	"video-streaming/internal/config"
)

type FrameCleanup struct {
	tmpDir string
	config *config.Config
}

func NewFrameCleanup(tmpDir string, cfg *config.Config) *FrameCleanup {
	return &FrameCleanup{
		tmpDir: tmpDir,
		config: cfg,
	}
}

func (fc *FrameCleanup) Start(ctx context.Context) {
	ticker := time.NewTicker(fc.config.CleanupInterval)
	defer ticker.Stop()

	log.Printf("Started cleanup service for: %s (window: %v)", fc.tmpDir, fc.config.CleanupWindow)

	for {
		select {
		case <-ctx.Done():
			log.Printf("Cleanup service stopped for: %s", fc.tmpDir)
			return
		case <-ticker.C:
			fc.cleanupOldFrames()
		}
	}
}

func (fc *FrameCleanup) cleanupOldFrames() {
	now := time.Now()
	cutoffTime := now.Add(-fc.config.CleanupWindow)
	deletedCount := 0

	files, err := filepath.Glob(filepath.Join(fc.tmpDir, "*.jpeg"))
	if err != nil {
		log.Printf("Error reading frames directory: %v", err)
		return
	}

	for _, file := range files {
		info, err := os.Stat(file)
		if err != nil {
			continue
		}

		if info.ModTime().Before(cutoffTime) {
			if err := os.Remove(file); err != nil {
				log.Printf("Failed to delete old frame %s: %v", file, err)
			} else {
				deletedCount++
			}
		}
	}

	if deletedCount > 0 {
		log.Printf("Cleaned up %d old frames from: %s", deletedCount, fc.tmpDir)
	}
}
