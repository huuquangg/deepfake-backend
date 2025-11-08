package repository

import (
	"context"
	"database/sql"
	"fmt"
	"time"

	"video-streaming/internal/dto"
)

type SessionRepository struct {
	db *sql.DB
}

func NewSessionRepository(db *sql.DB) *SessionRepository {
	return &SessionRepository{db: db}
}

// CreateSession inserts a new session into the database
func (r *SessionRepository) CreateSession(ctx context.Context, session *dto.SessionDTO) error {
	query := `
		INSERT INTO sessions (id, user_id, video_path, status, created_at, updated_at)
		VALUES ($1, $2, $3, $4, $5, $6)
	`
	_, err := r.db.ExecContext(
		ctx,
		query,
		session.ID,
		session.UserID,
		session.VideoPath,
		session.Status,
		session.CreatedAt,
		session.UpdatedAt,
	)
	if err != nil {
		return fmt.Errorf("failed to create session: %w", err)
	}
	return nil
}

// GetSessionByID retrieves a session by ID
func (r *SessionRepository) GetSessionByID(ctx context.Context, id string) (*dto.SessionDTO, error) {
	query := `
		SELECT id, user_id, video_path, status, created_at, updated_at
		FROM sessions
		WHERE id = $1
	`
	var session dto.SessionDTO
	err := r.db.QueryRowContext(ctx, query, id).Scan(
		&session.ID,
		&session.UserID,
		&session.VideoPath,
		&session.Status,
		&session.CreatedAt,
		&session.UpdatedAt,
	)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("session not found")
		}
		return nil, fmt.Errorf("failed to get session: %w", err)
	}
	return &session, nil
}

// UpdateSessionStatus updates the status of a session
func (r *SessionRepository) UpdateSessionStatus(ctx context.Context, id, status string) error {
	query := `
		UPDATE sessions
		SET status = $1, updated_at = $2
		WHERE id = $3
	`
	_, err := r.db.ExecContext(ctx, query, status, time.Now(), id)
	if err != nil {
		return fmt.Errorf("failed to update session status: %w", err)
	}
	return nil
}

// UpdateSessionVideoPath updates the video path of a session
func (r *SessionRepository) UpdateSessionVideoPath(ctx context.Context, id, videoPath string) error {
	query := `
		UPDATE sessions
		SET video_path = $1, updated_at = $2
		WHERE id = $3
	`
	_, err := r.db.ExecContext(ctx, query, videoPath, time.Now(), id)
	if err != nil {
		return fmt.Errorf("failed to update video path: %w", err)
	}
	return nil
}

// DeleteSession deletes a session by ID
func (r *SessionRepository) DeleteSession(ctx context.Context, id string) error {
	query := `DELETE FROM sessions WHERE id = $1`
	_, err := r.db.ExecContext(ctx, query, id)
	if err != nil {
		return fmt.Errorf("failed to delete session: %w", err)
	}
	return nil
}

// ListSessionsByUserID retrieves all sessions for a user
func (r *SessionRepository) ListSessionsByUserID(ctx context.Context, userID string) ([]*dto.SessionDTO, error) {
	query := `
		SELECT id, user_id, video_path, status, created_at, updated_at
		FROM sessions
		WHERE user_id = $1
		ORDER BY created_at DESC
	`
	rows, err := r.db.QueryContext(ctx, query, userID)
	if err != nil {
		return nil, fmt.Errorf("failed to list sessions: %w", err)
	}
	defer rows.Close()

	var sessions []*dto.SessionDTO
	for rows.Next() {
		var session dto.SessionDTO
		err := rows.Scan(
			&session.ID,
			&session.UserID,
			&session.VideoPath,
			&session.Status,
			&session.CreatedAt,
			&session.UpdatedAt,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to scan session: %w", err)
		}
		sessions = append(sessions, &session)
	}
	return sessions, nil
}
