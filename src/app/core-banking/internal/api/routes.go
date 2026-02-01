package api

import (
	"core-banking/internal/middleware"
	"net/http"
)

// SetupRoutes - Thiết lập các routes cho API
func SetupRoutes(authHandler *AuthHandler, accountHandler *AccountHandler, transactionHandler *TransactionHandler) *http.ServeMux {
	mux := http.NewServeMux()

	// AUTH ROUTES
	// Public routes (không cần JWT)
	mux.HandleFunc("/api/core-banking/auth/register", authHandler.Register)
	mux.HandleFunc("/api/core-banking/auth/login", authHandler.Login)

	// Protected routes (cần JWT)
	mux.Handle("/api/core-banking/auth/me", middleware.AuthMiddleware(http.HandlerFunc(authHandler.GetMe)))

	// ACCOUNT ROUTES 
	// Protected routes (cần JWT)
	mux.Handle("/api/core-banking/account/create", middleware.AuthMiddleware(http.HandlerFunc(accountHandler.CreateAccount)))
	mux.Handle("/api/core-banking/account/info", middleware.AuthMiddleware(http.HandlerFunc(accountHandler.GetMyAccount)))
	mux.Handle("/api/core-banking/account/balance", middleware.AuthMiddleware(http.HandlerFunc(accountHandler.GetBalance)))

	// TRANSACTION ROUTES 
	// Protected routes (cần JWT)
	mux.Handle("/api/core-banking/transaction/transfer", middleware.AuthMiddleware(http.HandlerFunc(transactionHandler.Transfer)))
	mux.Handle("/api/core-banking/transaction/history", middleware.AuthMiddleware(http.HandlerFunc(transactionHandler.GetTransactionHistory)))

	return mux
}