package repository

import (
	"core-banking/internal/models"
	"database/sql"
	"errors"
)

type TransactionRepository struct {
db *sql.DB
}

func NewTransactionRepository(db *sql.DB) *TransactionRepository {
return &TransactionRepository{db: db}
}

// CreateTransaction creates a new transaction 
func (r *TransactionRepository) CreateTransaction(tx *models.Transaction) error {
query := `
        INSERT INTO transactions (transaction_id, from_account_id, to_account_number, to_account_id, amount, description, status, created_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
        RETURNING id, created_at
    `

err := r.db.QueryRow(
query,
tx.TransactionID,
tx.FromAccountID,
tx.ToAccountNumber,
tx.ToAccountID, 
tx.Amount,
tx.Description,
tx.Status,
    ).Scan(&tx.ID, &tx.CreatedAt)

return err
}

// GetTransactionsByAccountID gets all transactions for an account (CẢ GỬI VÀ NHẬN)
func (r *TransactionRepository) GetTransactionsByAccountID(accountID int) ([]*models.Transaction, error) {
query := `
        SELECT id, transaction_id, from_account_id, to_account_number, to_account_id, amount, description, status, created_at, completed_at
        FROM transactions
        WHERE from_account_id = $1 OR to_account_id = $1
        ORDER BY created_at DESC
    `

rows, err := r.db.Query(query, accountID)
if err != nil {
return nil, err
    }
defer rows.Close()

var transactions []*models.Transaction
for rows.Next() {
tx := &models.Transaction{}
err := rows.Scan(
&tx.ID,
&tx.TransactionID,
&tx.FromAccountID,
&tx.ToAccountNumber,
&tx.ToAccountID, 
&tx.Amount,
&tx.Description,
&tx.Status,
&tx.CreatedAt,
&tx.CompletedAt,
        )
if err != nil {
return nil, err
        }
transactions = append(transactions, tx)
    }

return transactions, nil
}

// GetTransactionByID gets a transaction by ID
func (r *TransactionRepository) GetTransactionByID(id int) (*models.Transaction, error) {
query := `
        SELECT id, transaction_id, from_account_id, to_account_number, to_account_id, amount, description, status, created_at, completed_at
        FROM transactions
        WHERE id = $1
    `

tx := &models.Transaction{}
err := r.db.QueryRow(query, id).Scan(
&tx.ID,
&tx.TransactionID,
&tx.FromAccountID,
&tx.ToAccountNumber,
&tx.ToAccountID, 
&tx.Amount,
&tx.Description,
&tx.Status,
&tx.CreatedAt,
&tx.CompletedAt,
    )

if err == sql.ErrNoRows {
return nil, errors.New("transaction not found")
    }

return tx, err
}

// UpdateTransactionStatus updates transaction status
func (r *TransactionRepository) UpdateTransactionStatus(id int, status string) error {
query := `
        UPDATE transactions
        SET status = $1, completed_at = NOW()
        WHERE id = $2
    `

result, err := r.db.Exec(query, status, id)
if err != nil {
return err
    }

rowsAffected, err := result.RowsAffected()
if err != nil {
return err
    }

if rowsAffected == 0 {
return errors.New("transaction not found")
    }

return nil
}