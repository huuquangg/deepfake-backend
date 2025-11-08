package dto

import "time"

// SessionDTO represents session data for transfer
type SessionDTO struct {
	ID        string    `json:"id"`
	UserID    string    `json:"user_id"`
	VideoPath string    `json:"video_path"`
	Status    string    `json:"status"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}

// CreateSessionRequest represents request to create a session
type CreateSessionRequest struct {
	UserID string `json:"user_id"`
}

// CreateSessionResponse represents response after creating a session
type CreateSessionResponse struct {
	SessionID string `json:"session_id"`
	Status    string `json:"status"`
	CreatedAt string `json:"created_at"`
}

// UploadVideoResponse represents response after uploading video
type UploadVideoResponse struct {
	Message   string `json:"message"`
	SessionID string `json:"session_id"`
	Status    string `json:"status"`
}
