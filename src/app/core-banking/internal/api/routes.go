package api

import (
	"net/http"

	"core-banking/internal/middleware"
)

// SetupRoutes - Thiết lập các routes cho API
func SetupRoutes(authHandler *AuthHandler) *http.ServeMux {
    mux := http.NewServeMux()
    
    // Public routes (không cần JWT)
    mux.HandleFunc("/api/auth/register", authHandler.Register)
    mux.HandleFunc("/api/auth/login", authHandler.Login)
    
    // Protected routes (cần JWT)
    mux.Handle("/api/auth/me", middleware.AuthMiddleware(http.HandlerFunc(authHandler.GetMe)))
    
    return mux
}