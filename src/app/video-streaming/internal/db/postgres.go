package db

import (
	"database/sql"
	"fmt"
	"log"

	"video-streaming/internal/config"

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
		// Create sessions table
		`CREATE TABLE IF NOT EXISTS sessions (
			id TEXT PRIMARY KEY,
			user_id TEXT NOT NULL,
			video_path TEXT,
			status TEXT NOT NULL,
			created_at TIMESTAMP WITH TIME ZONE NOT NULL,
			updated_at TIMESTAMP WITH TIME ZONE NOT NULL
		)`,

		// Create indexes for sessions
		`CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id)`,
		`CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status)`,
		`CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON sessions(created_at DESC)`,

		// Create frames table
		`CREATE TABLE IF NOT EXISTS frames (
			id TEXT PRIMARY KEY,
			session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
			filename TEXT NOT NULL,
			file_path TEXT NOT NULL,
			created_at TIMESTAMP WITH TIME ZONE NOT NULL,
			UNIQUE(session_id, filename)
		)`,

		// Create index for frames
		`CREATE INDEX IF NOT EXISTS idx_frames_session_id ON frames(session_id)`,
	}

	for i, migration := range migrations {
		if _, err := db.Exec(migration); err != nil {
			return fmt.Errorf("migration %d failed: %w", i+1, err)
		}
	}

	log.Printf("Migrations completed successfully in schema: %s", schema)
	return nil
}
