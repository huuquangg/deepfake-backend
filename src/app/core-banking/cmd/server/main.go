package main

import (
	"log"
	"net/http"

	"core-banking/internal/api"
	"core-banking/internal/config"
	"core-banking/internal/db"
	"core-banking/internal/repository"
	"core-banking/internal/service"
)

func main() {
	log.Println("core-banking server is starting...")

	// Load configuration
	cfg := config.New()

	// Connect to PostgreSQL
	database, err := db.ConnectPostgres(cfg)
	if err != nil {
		log.Fatalf("Failed to connect to database: %v", err)
	}
	defer database.Close()
	log.Println("Database connected successfully")

	// Initialize Repository layer
	userRepo := repository.NewUserRepository(database)
	accountRepo := repository.NewAccountRepository(database)
	transactionRepo := repository.NewTransactionRepository(database)

	// Initialize Service layer
	accountService := service.NewAccountService(accountRepo)
	authService := service.NewAuthService(userRepo, accountRepo)
	transactionService := service.NewTransactionService(transactionRepo, accountRepo)

	// Initialize Handler layer
	authHandler := api.NewAuthHandler(authService)
	accountHandler := api.NewAccountHandler(accountService)
	transactionHandler := api.NewTransactionHandler(transactionService)

	// Setup routes
	mux := api.SetupRoutes(authHandler, accountHandler, transactionHandler)

	// Configure HTTP server
	server := &http.Server{
		Addr:         cfg.ServerAddress,
		Handler:      mux,
		ReadTimeout:  cfg.ReadTimeout,
		WriteTimeout: cfg.WriteTimeout,
	}

	// Start server
	log.Printf("Server starting on %s", cfg.ServerAddress)
	if err := server.ListenAndServe(); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}