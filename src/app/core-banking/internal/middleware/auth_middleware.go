package middleware

import (
	"context"
	"net/http"
	"strings"

	"core-banking/internal/utils"
)

type contextKey string

const UserIDKey contextKey = "userID"

// AuthMiddleware - Kiểm tra JWT token
func AuthMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        // 1. Lấy token từ header Authorization
        authHeader := r.Header.Get("Authorization")
        if authHeader == "" {
            http.Error(w, "Missing authorization header", http.StatusUnauthorized)
            return
        }
        
        // 2. Token format: "Bearer <token>"
        parts := strings.Split(authHeader, " ")
        if len(parts) != 2 || parts[0] != "Bearer" {
            http.Error(w, "Invalid authorization header format", http.StatusUnauthorized)
            return
        }
        
        token := parts[1]
        
        // 3. Validate token
        userID, err := utils.ParseToken(token)
        if err != nil {
            http.Error(w, "Invalid or expired token", http.StatusUnauthorized)
            return
        }
        
        // 4. Lưu userID vào context để dùng ở handler
        ctx := context.WithValue(r.Context(), UserIDKey, userID)
        
        // 5. Gọi handler tiếp theo
        next.ServeHTTP(w, r.WithContext(ctx))
    })
}
