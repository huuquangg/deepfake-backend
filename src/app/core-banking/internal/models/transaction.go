package models

import "time"

type Transaction struct {
	ID              int       `json:"id"`
	TransactionID   string    `json:"transaction_id"`
	FromAccountID   int       `json:"from_account_id"`
	ToAccountNumber string    `json:"to_account_number"`
	Amount          float64   `json:"amount"`
	Description     string    `json:"description"`
	Status          string    `json:"status"`
	CreatedAt       time.Time `json:"created_at"`
	CompletedAt     *time.Time `json:"completed_at,omitempty"`
}

// TransferRequest is the payload for money transfer
type TransferRequest struct {
	ToAccountNumber string  `json:"to_account_number"`
	Amount          float64 `json:"amount"`
	Description     string  `json:"description"`
	FaceImageBase64 string  `json:"face_image_base64"` // For deepfake detection
}

// TransferResponse is the response after transfer
type TransferResponse struct {
	Transaction Transaction `json:"transaction"`
	Message     string      `json:"message"`
	Success     bool        `json:"success"`
}