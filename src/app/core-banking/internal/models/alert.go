package models

import "time"

type Alert struct {
	ID            int       `json:"id"`
	AlertID       string    `json:"alert_id"`
	TransactionID *int      `json:"transaction_id,omitempty"`
	UserID        int       `json:"user_id"`
	DeepfakeScore float64   `json:"deepfake_score"`
	AlertType     string    `json:"alert_type"`
	Status        string    `json:"status"`
	Location      string    `json:"location,omitempty"`
	CreatedAt     time.Time `json:"created_at"`
}