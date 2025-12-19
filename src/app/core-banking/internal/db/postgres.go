package db

import (
	"database/sql"
	"fmt"
	"log"

	"core-banking/internal/config"

	_ "github.com/lib/pq"
)

func ConnectPostgres(cfg *config.Config) (*sql.DB, error) {
	dsn := fmt.Sprintf(
		"host=%s port=%d user=%s password=%s dbname=%s sslmode=%s",
		cfg.PostgresHost,
		cfg.PostgresPort,
		cfg.PostgresUser,
		cfg.PostgresPassword,
		cfg.PostgresDB,
		cfg.PostgresSSLMode,
	)

	db, err := sql.Open("postgres", dsn)
	if err != nil {
		return nil, fmt.Errorf("failed to open database: %w", err)
	}

	// Test connection
	if err := db.Ping(); err != nil {
		return nil, fmt.Errorf("failed to ping database: %w", err)
	}

	// Create schema if it doesn't exist
	createSchemaSQL := fmt.Sprintf("CREATE SCHEMA IF NOT EXISTS %s", cfg.PostgresSchema)
	if _, err := db.Exec(createSchemaSQL); err != nil {
		return nil, fmt.Errorf("failed to create schema: %w", err)
	}

	// Set search_path to use the schema
	setSearchPathSQL := fmt.Sprintf("SET search_path TO %s, public", cfg.PostgresSchema)
	if _, err := db.Exec(setSearchPathSQL); err != nil {
		return nil, fmt.Errorf("failed to set search_path: %w", err)
	}

	// Run migrations
	if err := runMigrations(db, cfg.PostgresSchema); err != nil {
		return nil, fmt.Errorf("failed to run migrations: %w", err)
	}

	// Set connection pool settings
	db.SetMaxOpenConns(25)
	db.SetMaxIdleConns(5)

	log.Printf("PostgreSQL connection established (database: %s, schema: %s)", cfg.PostgresDB, cfg.PostgresSchema)
	return db, nil
}

func runMigrations(db *sql.DB, schema string) error {
	log.Println("Running migrations...")

	migrations := []string{
		// Create users table
		`CREATE TABLE IF NOT EXISTS users (
			id SERIAL PRIMARY KEY,
			username VARCHAR(50) UNIQUE NOT NULL,
			email VARCHAR(100) UNIQUE NOT NULL,
			password_hash VARCHAR(255) NOT NULL,
			full_name VARCHAR(100) NOT NULL,
			phone VARCHAR(20),
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
		)`,

		// Create indexes for users
		`CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)`,
		`CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)`,

		// Create accounts table
		`CREATE TABLE IF NOT EXISTS accounts (
			id SERIAL PRIMARY KEY,
			user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
			account_number VARCHAR(20) UNIQUE NOT NULL,
			balance DECIMAL(15, 2) DEFAULT 0.00,
			account_type VARCHAR(20) DEFAULT 'savings',
			status VARCHAR(20) DEFAULT 'active',
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
		)`,

		// Create indexes for accounts
		`CREATE INDEX IF NOT EXISTS idx_accounts_user_id ON accounts(user_id)`,
		`CREATE INDEX IF NOT EXISTS idx_accounts_account_number ON accounts(account_number)`,

		// Create transactions table
		`CREATE TABLE IF NOT EXISTS transactions (
			id SERIAL PRIMARY KEY,
			transaction_id VARCHAR(50) UNIQUE NOT NULL,
			from_account_id INTEGER NOT NULL REFERENCES accounts(id),
			to_account_number VARCHAR(20) NOT NULL,
			amount DECIMAL(15, 2) NOT NULL,
			description TEXT,
			status VARCHAR(20) DEFAULT 'pending',
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			completed_at TIMESTAMP
		)`,

		// Create indexes for transactions
		`CREATE INDEX IF NOT EXISTS idx_transactions_from_account ON transactions(from_account_id)`,
		`CREATE INDEX IF NOT EXISTS idx_transactions_to_account ON transactions(to_account_number)`,
		`CREATE INDEX IF NOT EXISTS idx_transactions_status ON transactions(status)`,
		`CREATE INDEX IF NOT EXISTS idx_transactions_created_at ON transactions(created_at)`,

		// Create alerts table
		`CREATE TABLE IF NOT EXISTS alerts (
			id SERIAL PRIMARY KEY,
			user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
			transaction_id INTEGER REFERENCES transactions(id),
			alert_type VARCHAR(50) NOT NULL,
			message TEXT NOT NULL,
			is_read BOOLEAN DEFAULT FALSE,
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
		)`,

		// Create index for alerts
		`CREATE INDEX IF NOT EXISTS idx_alerts_user_id ON alerts(user_id)`,
		`CREATE INDEX IF NOT EXISTS idx_alerts_is_read ON alerts(is_read)`,
	}

	for i, migration := range migrations {
		if _, err := db.Exec(migration); err != nil {
			return fmt.Errorf("migration %d failed: %w", i+1, err)
		}
	}

	log.Printf("Migrations completed successfully in schema: %s", schema)
	return nil
}
