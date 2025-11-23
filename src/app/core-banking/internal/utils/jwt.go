package utils

import (
	"errors"
	"time"

	"github.com/golang-jwt/jwt/v5"
)

var jwtSecret = []byte("deepfake-bank-jwt-secret-2024-secure-key-x9Yz4")

type Claims struct {
    UserID int `json:"user_id"`
    jwt.RegisteredClaims
}

// GenerateToken - Tạo JWT token cho user
func GenerateToken(userID int) (string, error) {
    expirationTime := time.Now().Add(24 * time.Hour) // Token hết hạn sau 24h
    
    claims := &Claims{
        UserID: userID,
        RegisteredClaims: jwt.RegisteredClaims{
            ExpiresAt: jwt.NewNumericDate(expirationTime),
            IssuedAt:  jwt.NewNumericDate(time.Now()),
        },
    }
    
    token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
    return token.SignedString(jwtSecret)
}

// ValidateToken - Kiểm tra JWT token có hợp lệ không
func ValidateToken(tokenString string) (*Claims, error) {
    claims := &Claims{}
    
    token, err := jwt.ParseWithClaims(tokenString, claims, func(token *jwt.Token) (interface{}, error) {
        return jwtSecret, nil
    })
    
    if err != nil {
        return nil, err
    }
    
    if !token.Valid {
        return nil, errors.New("invalid token")
    }
    
    return claims, nil
}

// ParseToken - Lấy userID từ token
func ParseToken(tokenString string) (int, error) {
    claims, err := ValidateToken(tokenString)
    if err != nil {
        return 0, err
    }
    return claims.UserID, nil
}