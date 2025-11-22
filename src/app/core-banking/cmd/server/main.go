package main

import (
	"database/sql"
	"fmt"
	"log"
	"net/http"
	"os"

	"core-banking/internal/api"
	"core-banking/internal/repository"
	"core-banking/internal/service"

	"github.com/joho/godotenv"
	_ "github.com/lib/pq"
)

func main() {
    // 1. Load .env file
    if err := godotenv.Load("/Applications/Tien/deepfakeFrontendBackend/deepfake-backend/.env"); err != nil {
        log.Println("Warning: .env file not found")
    }
    
    // 2. Kết nối database
    dbHost := os.Getenv("POSTGRES_HOST")
	dbPort := os.Getenv("POSTGRES_PORT")
	dbUser := os.Getenv("POSTGRES_USER")
	dbPassword := os.Getenv("POSTGRES_PASSWORD")
	dbName := os.Getenv("POSTGRES_DB")
    
    connStr := fmt.Sprintf("host=%s port=%s user=%s password=%s dbname=%s sslmode=disable",
        dbHost, dbPort, dbUser, dbPassword, dbName)
    
    db, err := sql.Open("postgres", connStr)
    if err != nil {
        log.Fatal("Failed to connect to database:", err)
    }
    defer db.Close()
    
    // Test connection
    if err := db.Ping(); err != nil {
        log.Fatal("Failed to ping database:", err)
    }
    log.Println("✅ Connected to database successfully!")
    
    // 3. Khởi tạo layers
    userRepo := repository.NewUserRepository(db)
    authService := service.NewAuthService(userRepo)
    authHandler := api.NewAuthHandler(authService)
    
    // 4. Setup routes
    mux := api.SetupRoutes(authHandler)
    
    // 5. Start server
    port := os.Getenv("SERVER_PORT")
    if port == "" {
        port = "8085"
    }
    
    log.Printf("🚀 Server is running on port %s\n", port)
    if err := http.ListenAndServe(":"+port, mux); err != nil {
        log.Fatal("Failed to start server:", err)
    }
}