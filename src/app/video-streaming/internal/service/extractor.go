package service

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"time"

	"video-streaming/internal/config"

	"github.com/google/uuid"
)

type FrameExtractor struct {
	videoPath string
	outputDir string
	config    *config.Config
}

func NewFrameExtractor(videoPath, outputDir string, cfg *config.Config) *FrameExtractor {
	return &FrameExtractor{
		videoPath: videoPath,
		outputDir: outputDir,
		config:    cfg,
	}
}

func (fe *FrameExtractor) Start(ctx context.Context) error {
	log.Printf("Starting frame extraction from: %s", fe.videoPath)

	tempPattern := filepath.Join(fe.outputDir, "frame_%06d.jpeg")

	cmd := exec.CommandContext(ctx,
		"ffmpeg",
		"-i", fe.videoPath,
		"-vf", fmt.Sprintf("fps=%d", fe.config.FrameRate),
		"-q:v", fmt.Sprintf("%d", fe.config.JPEGQuality),
		"-f", "image2",
		tempPattern,
	)

	// Capture output for debugging
	cmd.Stderr = os.Stderr

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to start ffmpeg: %w", err)
	}

	// Rename frames to UUID in background
	go fe.renameFrames(ctx)

	// Wait for completion
	err := cmd.Wait()
	if err != nil && ctx.Err() == nil {
		return fmt.Errorf("ffmpeg error: %w", err)
	}

	log.Printf("Frame extraction completed for: %s", fe.videoPath)
	return nil
}

func (fe *FrameExtractor) renameFrames(ctx context.Context) {
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	frameIndex := 1

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			oldPath := filepath.Join(fe.outputDir, fmt.Sprintf("frame_%06d.jpeg", frameIndex))

			if _, err := os.Stat(oldPath); err == nil {
				newFilename := uuid.New().String() + ".jpeg"
				newPath := filepath.Join(fe.outputDir, newFilename)

				if err := os.Rename(oldPath, newPath); err != nil {
					log.Printf("Failed to rename frame: %v", err)
				} else {
					log.Printf("Extracted frame: %s", newFilename)
				}

				frameIndex++
			}
		}
	}
}
