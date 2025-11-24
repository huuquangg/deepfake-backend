package service

import (
	"errors"

	"core-banking/internal/dto"
	"core-banking/internal/models"
	"core-banking/internal/repository"
	"core-banking/internal/utils"
)

type AuthService struct {
    userRepo *repository.UserRepository
}

// NewAuthService - Tạo auth service mới
func NewAuthService(userRepo *repository.UserRepository) *AuthService {
    return &AuthService{
        userRepo: userRepo,
    }
}

// RegisterRequest - Dữ liệu đăng ký
type RegisterRequest struct {
    Username string `json:"username"`
    Email    string `json:"email"`
    Password string `json:"password"`
    FullName string `json:"full_name"`
    Phone    string `json:"phone"`
}

// LoginRequest - Dữ liệu đăng nhập
type LoginRequest struct {
    Username string `json:"username"`
    Password string `json:"password"`
}

// AuthResponse - Kết quả trả về sau khi login/register
type AuthResponse struct {
    Token string       `json:"token"`
    User  *dto.UserDTO `json:"user"`
}

// Register - Đăng ký user mới
func (s *AuthService) Register(req *RegisterRequest) (*AuthResponse, error) {
    // 1. Kiểm tra username đã tồn tại chưa
    existingUser, err := s.userRepo.GetUserByUsername(req.Username)
    if err != nil {
        return nil, err
    }
    if existingUser != nil {
        return nil, errors.New("username already exists")
    }
    
    // 2. Kiểm tra email đã tồn tại chưa
    existingEmail, err := s.userRepo.GetUserByEmail(req.Email)
    if err != nil {
        return nil, err
    }
    if existingEmail != nil {
        return nil, errors.New("email already exists")
    }
    
    // 3. Hash password
    hashedPassword, err := utils.HashPassword(req.Password)
    if err != nil {
        return nil, err
    }
    
    // 4. Tạo user mới
    user := &models.User{
        Username:     req.Username,
        Email:        req.Email,
        PasswordHash: hashedPassword,
        FullName:     req.FullName,
        Phone:        req.Phone,
    }
    
    // 5. Lưu vào database
    err = s.userRepo.CreateUser(user)
    if err != nil {
        return nil, err
    }
    
    // 6. Tạo JWT token
    token, err := utils.GenerateToken(user.ID)
    if err != nil {
        return nil, err
    }
    // 7. Convert sang DTO và trả về kết quả
    userDTO := dto.ToUserDTO(user)
    
    return &AuthResponse{
        Token: token,
        User:  userDTO,
    }, nil
}

// Login - Đăng nhập
func (s *AuthService) Login(req *LoginRequest) (*AuthResponse, error) {
    // 1. Tìm user theo username
    user, err := s.userRepo.GetUserByUsername(req.Username)
    if err != nil {
        return nil, err
    }
    if user == nil {
        return nil, errors.New("invalid username or password")
    }
    
    // 2. So sánh password
    err = utils.ComparePassword(user.PasswordHash, req.Password)
    if err != nil {
        return nil, errors.New("invalid username or password")
    }
    
    // 3. Tạo JWT token
    token, err := utils.GenerateToken(user.ID)
    if err != nil {
        return nil, err
    }
   // 7. Convert sang DTO và trả về kết quả
    userDTO := dto.ToUserDTO(user)
    
    return &AuthResponse{
        Token: token,
        User:  userDTO,
    }, nil
}

 