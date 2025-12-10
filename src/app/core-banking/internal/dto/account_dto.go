package dto

import (
	"core-banking/internal/models"
	"time"
)

// AccountDTO represents account data for API responses
type AccountDTO struct {
	ID            int       `json:"id"`
	UserID        int       `json:"user_id"`
	AccountNumber string    `json:"account_number"`
	Balance       float64   `json:"balance"`
	AccountType   string    `json:"account_type"`
	Status        string    `json:"status"`
	CreatedAt     time.Time `json:"created_at"`
	UpdatedAt     time.Time `json:"updated_at"`
}

// ToAccountDTO converts Account model to AccountDTO
func ToAccountDTO(account *models.Account) *AccountDTO {
	return &AccountDTO{
		ID:            account.ID,
		UserID:        account.UserID,
		AccountNumber: account.AccountNumber,
		Balance:       account.Balance,
		AccountType:   account.AccountType,
		Status:        account.Status,
		CreatedAt:     account.CreatedAt,
		UpdatedAt:     account.UpdatedAt,
	}
}