package ffmpeg

import (
	"context"
	"fmt"
	"os/exec"
)

// CheckInstallation verifies if FFmpeg is installed and accessible
func CheckInstallation() error {
	cmd := exec.Command("ffmpeg", "-version")
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("ffmpeg is not installed or not in PATH: %w", err)
	}
	return nil
}

// GetVideoInfo retrieves basic video information
func GetVideoInfo(ctx context.Context, videoPath string) (string, error) {
	cmd := exec.CommandContext(ctx, "ffprobe",
		"-v", "error",
		"-show_entries", "format=duration",
		"-of", "default=noprint_wrappers=1:nokey=1",
		videoPath,
	)

	output, err := cmd.Output()
	if err != nil {
		return "", fmt.Errorf("failed to get video info: %w", err)
	}

	return string(output), nil
}

// GetVideoMetadata retrieves detailed video metadata
func GetVideoMetadata(ctx context.Context, videoPath string) (map[string]string, error) {
	cmd := exec.CommandContext(ctx, "ffprobe",
		"-v", "error",
		"-show_entries", "format=duration,size,bit_rate",
		"-show_entries", "stream=codec_name,width,height,r_frame_rate",
		"-of", "json",
		videoPath,
	)

	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("failed to get video metadata: %w", err)
	}

	// Simple return for now, can be parsed into struct
	return map[string]string{"raw": string(output)}, nil
}
