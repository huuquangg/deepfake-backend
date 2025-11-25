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

// TransactionRequest is the payload for transactions (transfer, withdraw, deposit)
type TransactionRequest struct {
    Type            string  `json:"type"`              // "TRANSFER", "WITHDRAW", "DEPOSIT"
    ToAccountNumber string  `json:"to_account_number"` // For TRANSFER only
    Amount          float64 `json:"amount"`
    Description     string  `json:"description"`
    FaceImageBase64 string  `json:"face_image_base64"` // For deepfake detection
}


// TransferResponse is the response after transfer
type TransactionResponse struct {
	Transaction Transaction `json:"transaction"`
	Message     string      `json:"message"`
	Success     bool        `json:"success"`
}