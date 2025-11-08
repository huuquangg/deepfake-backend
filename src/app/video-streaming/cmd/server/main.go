package main

import (
	"context"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"video-streaming/internal/api"
	"video-streaming/internal/config"
	"video-streaming/internal/db"
	"video-streaming/internal/repository"
	"video-streaming/internal/service"
)

func main() {
	log.Println("Starting Video Streaming...")

	// Load configuration
	cfg := config.New()

	// Create base tmp directory
	if err := os.MkdirAll(cfg.TmpDir, 0755); err != nil {
		log.Fatalf("Failed to create tmp directory: %v", err)
	}

	// Connect to PostgreSQL
	dbConn, err := db.ConnectPostgres(cfg)
	if err != nil {
		log.Fatalf("Failed to connect to database: %v", err)
	}
	defer dbConn.Close()
	log.Println("Database connected successfully")

	// Initialize repository
	sessionRepo := repository.NewSessionRepository(dbConn)

	// Initialize services
	sessionService := service.NewSessionService(cfg, sessionRepo)

	// Setup HTTP server
	handler := api.NewHandler(sessionService)
	router := api.SetupRoutes(handler)
	server := api.NewHTTPServer(cfg, router)

	// Start server in goroutine
	go func() {
		log.Printf("Server starting on %s", cfg.ServerAddress)
		if err := server.ListenAndServe(); err != nil {
			log.Fatalf("Server failed: %v", err)
		}
	}()

	// Wait for interrupt signal
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Println("Shutting down server...")

	// Graceful shutdown
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	sessionService.Shutdown()
	if err := server.Shutdown(ctx); err != nil {
		log.Fatalf("Server forced to shutdown: %v", err)
	}

	log.Println("Server exited gracefully")
}
