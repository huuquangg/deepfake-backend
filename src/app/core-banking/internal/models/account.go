package models

import "time"

type Account struct {
	ID            int       `json:"id"`
	UserID        int       `json:"user_id"`
	AccountNumber string    `json:"account_number"`
	Balance       float64   `json:"balance"`
	AccountType   string    `json:"account_type"`
	Status        string    `json:"status"`
	CreatedAt     time.Time `json:"created_at"`
	UpdatedAt     time.Time `json:"updated_at"`
}