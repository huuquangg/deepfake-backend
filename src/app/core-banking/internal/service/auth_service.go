package service

import (
	"errors"
	"fmt"
	"log"
	"math/rand"
	"time"

	"core-banking/internal/dto"
	"core-banking/internal/models"
	"core-banking/internal/repository"
	"core-banking/internal/utils"
)

type AuthService struct {
userRepo    *repository.UserRepository
accountRepo *repository.AccountRepository
}

// NewAuthService - Tạo auth service mới
func NewAuthService(userRepo *repository.UserRepository, accountRepo *repository.AccountRepository) *AuthService {
// Seed random number generator
rand.Seed(time.Now().UnixNano())
return &AuthService{
userRepo:    userRepo,
accountRepo: accountRepo,
    }
}

// generateAccountNumber - Tạo số tài khoản ngẫu nhiên 10 chữ số
func generateAccountNumber() string {
return fmt.Sprintf("%010d", rand.Intn(10000000000))
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

// 6. TỰ ĐỘNG TẠO ACCOUNT cho user mới
accountNumber := generateAccountNumber()
account := &models.Account{
UserID:        user.ID,
AccountNumber: accountNumber,
Balance:       0.0,
AccountType:   "SAVINGS",
Status:        "ACTIVE",
    }

err = s.accountRepo.CreateAccount(account)
if err != nil {
// Log error nhưng không fail registration
log.Printf(" Failed to create account for user %d: %v", user.ID, err)
// Có thể return error hoặc tiếp tục tùy theo logic nghiệp vụ
// Ở đây tôi cho phép user đăng ký thành công dù account tạo fail
    } else {
log.Printf(" Auto-created account %s for user %d", accountNumber, user.ID)
    }

// 7. Tạo JWT token
token, err := utils.GenerateToken(user.ID)
if err != nil {
return nil, err
    }

// 8. Convert sang DTO và trả về kết quả
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

// 3. KIỂM TRA VÀ TẠO ACCOUNT NẾU CHƯA CÓ (cho user cũ)
_, err = s.accountRepo.GetAccountByUserID(user.ID)
if err != nil {
// Account chưa tồn tại, tạo mới
log.Printf(" User %d doesn't have account, creating...", user.ID)
accountNumber := generateAccountNumber()
account := &models.Account{
UserID:        user.ID,
AccountNumber: accountNumber,
Balance:       0.0,
AccountType:   "SAVINGS",
Status:        "ACTIVE",
        }
err = s.accountRepo.CreateAccount(account)
if err != nil {
log.Printf("Failed to create account for user %d: %v", user.ID, err)
        } else {
log.Printf("Auto-created account %s for user %d", accountNumber, user.ID)
        }
    }

// 4. Tạo JWT token
token, err := utils.GenerateToken(user.ID)
if err != nil {
return nil, err
    }

// 5. Convert sang DTO và trả về kết quả
userDTO := dto.ToUserDTO(user)
return &AuthResponse{
Token: token,
User:  userDTO,
    }, nil
}