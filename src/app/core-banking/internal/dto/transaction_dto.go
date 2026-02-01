package dto

import (
	"core-banking/internal/models"
	"time"
)

// TransactionDTO represents transaction data for API responses
type TransactionDTO struct {
	ID              int        `json:"id"`
	TransactionID   string     `json:"transaction_id"`
	FromAccountID   int        `json:"from_account_id"`
	ToAccountNumber string     `json:"to_account_number"`
	Amount          float64    `json:"amount"`
	Description     string     `json:"description"`
	Status          string     `json:"status"`
	CreatedAt       time.Time  `json:"created_at"`
	CompletedAt     *time.Time `json:"completed_at,omitempty"`
}

// ToTransactionDTO converts Transaction model to TransactionDTO
func ToTransactionDTO(tx *models.Transaction) *TransactionDTO {
	return &TransactionDTO{
		ID:              tx.ID,
		TransactionID:   tx.TransactionID,
		FromAccountID:   tx.FromAccountID,
		ToAccountNumber: tx.ToAccountNumber,
		Amount:          tx.Amount,
		Description:     tx.Description,
		Status:          tx.Status,
		CreatedAt:       tx.CreatedAt,
		CompletedAt:     tx.CompletedAt,
	}
}

// ToTransactionDTOList converts slice of Transactions to slice of TransactionDTOs
func ToTransactionDTOList(transactions []*models.Transaction) []*TransactionDTO {
	dtos := make([]*TransactionDTO, len(transactions))
	for i, tx := range transactions {
		dtos[i] = ToTransactionDTO(tx)
	}
	return dtos
}

// TransactionResponse is the response after a transaction operation
type TransactionResponse struct {
	Transaction *TransactionDTO `json:"transaction"`
	Message     string          `json:"message"`
	Success     bool            `json:"success"`
}