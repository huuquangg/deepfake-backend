package repository

import (
	"database/sql"
	"time"

	"core-banking/internal/models"
)

type UserRepository struct {
    db *sql.DB
}

// NewUserRepository - Tạo repository mới
func NewUserRepository(db *sql.DB) *UserRepository {
    return &UserRepository{db: db}
}

// CreateUser - Tạo user mới trong database
func (r *UserRepository) CreateUser(user *models.User) error {
    query := `
        INSERT INTO users (username, email, password_hash, full_name, phone, created_at, updated_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        RETURNING id
    `
    
    now := time.Now()
    err := r.db.QueryRow(
        query,
        user.Username,
        user.Email,
        user.PasswordHash,
        user.FullName,
        user.Phone,
        now,
        now,
    ).Scan(&user.ID)
    
    return err
}

// GetUserByUsername - Tìm user theo username
func (r *UserRepository) GetUserByUsername(username string) (*models.User, error) {
    query := `
        SELECT id, username, email, password_hash, full_name, phone, created_at, updated_at
        FROM users
        WHERE username = $1
    `
    
    user := &models.User{}
    err := r.db.QueryRow(query, username).Scan(
        &user.ID,
        &user.Username,
        &user.Email,
        &user.PasswordHash,
        &user.FullName,
        &user.Phone,
        &user.CreatedAt,
        &user.UpdatedAt,
    )
    
    if err == sql.ErrNoRows {
        return nil, nil // Không tìm thấy user
    }
    
    return user, err
}

// GetUserByID - Tìm user theo ID
func (r *UserRepository) GetUserByID(id int) (*models.User, error) {
    query := `
        SELECT id, username, email, password_hash, full_name, phone, created_at, updated_at
        FROM users
        WHERE id = $1
    `
    
    user := &models.User{}
    err := r.db.QueryRow(query, id).Scan(
        &user.ID,
        &user.Username,
        &user.Email,
        &user.PasswordHash,
        &user.FullName,
        &user.Phone,
        &user.CreatedAt,
        &user.UpdatedAt,
    )
    
    if err == sql.ErrNoRows {
        return nil, nil // Không tìm thấy user
    }
    
    return user, err
}

// GetUserByEmail - Tìm user theo email
func (r *UserRepository) GetUserByEmail(email string) (*models.User, error) {
    query := `
        SELECT id, username, email, password_hash, full_name, phone, created_at, updated_at
        FROM users
        WHERE email = $1
    `
    
    user := &models.User{}
    err := r.db.QueryRow(query, email).Scan(
        &user.ID,
        &user.Username,
        &user.Email,
        &user.PasswordHash,
        &user.FullName,
        &user.Phone,
        &user.CreatedAt,
        &user.UpdatedAt,
    )
    
    if err == sql.ErrNoRows {
        return nil, nil // Không tìm thấy user
    }
    
    return user, err
}