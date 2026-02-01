package repository

import (
	"core-banking/internal/models"
	"database/sql"
	"errors"
)

type AccountRepository struct {
	db *sql.DB
}

func NewAccountRepository(db *sql.DB) *AccountRepository {
	return &AccountRepository{db: db}
}

// CreateAccount creates a new account for a user
func (r *AccountRepository) CreateAccount(account *models.Account) error {
	query := `
		INSERT INTO accounts (user_id, account_number, balance, account_type, status, created_at, updated_at)
		VALUES ($1, $2, $3, $4, $5, NOW(), NOW())
		RETURNING id, created_at, updated_at
	`
	
	err := r.db.QueryRow(
		query,
		account.UserID,
		account.AccountNumber,
		account.Balance,
		account.AccountType,
		account.Status,
	).Scan(&account.ID, &account.CreatedAt, &account.UpdatedAt)
	
	return err
}

// GetAccountByUserID gets account by user ID
func (r *AccountRepository) GetAccountByUserID(userID int) (*models.Account, error) {
	query := `
		SELECT id, user_id, account_number, balance, account_type, status, created_at, updated_at
		FROM accounts
		WHERE user_id = $1
	`
	
	account := &models.Account{}
	err := r.db.QueryRow(query, userID).Scan(
		&account.ID,
		&account.UserID,
		&account.AccountNumber,
		&account.Balance,
		&account.AccountType,
		&account.Status,
		&account.CreatedAt,
		&account.UpdatedAt,
	)
	
	if err == sql.ErrNoRows {
		return nil, errors.New("account not found")
	}
	
	return account, err
}

// GetAccountByNumber gets account by account number
func (r *AccountRepository) GetAccountByNumber(accountNumber string) (*models.Account, error) {
	query := `
		SELECT id, user_id, account_number, balance, account_type, status, created_at, updated_at
		FROM accounts
		WHERE account_number = $1
	`
	
	account := &models.Account{}
	err := r.db.QueryRow(query, accountNumber).Scan(
		&account.ID,
		&account.UserID,
		&account.AccountNumber,
		&account.Balance,
		&account.AccountType,
		&account.Status,
		&account.CreatedAt,
		&account.UpdatedAt,
	)
	
	if err == sql.ErrNoRows {
		return nil, errors.New("account not found")
	}
	
	return account, err
}

// UpdateBalance updates account balance
func (r *AccountRepository) UpdateBalance(accountID int, newBalance float64) error {
	query := `
		UPDATE accounts
		SET balance = $1, updated_at = NOW()
		WHERE id = $2
	`
	
	result, err := r.db.Exec(query, newBalance, accountID)
	if err != nil {
		return err
	}
	
	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return err
	}
	
	if rowsAffected == 0 {
		return errors.New("account not found")
	}
	
	return nil
}