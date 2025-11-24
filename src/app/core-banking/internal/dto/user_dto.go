package dto

import (
	"time"

	"core-banking/internal/models"
)

// UserDTO - Dữ liệu user trả về cho client (KHÔNG có password_hash)
type UserDTO struct {
    ID        int       `json:"id"`
    Username  string    `json:"username"`
    Email     string    `json:"email"`
    FullName  string    `json:"full_name"`
    Phone     string    `json:"phone"`
    CreatedAt time.Time `json:"created_at"`
    UpdatedAt time.Time `json:"updated_at"`
}

// ToUserDTO - Convert từ models.User sang UserDTO (loại bỏ password_hash)
func ToUserDTO(user *models.User) *UserDTO {
    return &UserDTO{
        ID:        user.ID,
        Username:  user.Username,
        Email:     user.Email,
        FullName:  user.FullName,
        Phone:     user.Phone,
        CreatedAt: user.CreatedAt,
        UpdatedAt: user.UpdatedAt,
    }
}