package main

import (
	"context"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"video-streaming/internal/aggregator"
	"video-streaming/internal/api"
	"video-streaming/internal/config"
	"video-streaming/internal/db"
	repository "video-streaming/internal/repository"
	service "video-streaming/internal/service"
)

func main() {
	log.Println("video-streaming server is starting...")

	// Load configuration
	configuration := config.New()

	// Connect to PostgreSQL
	dbConnector, err := db.ConnectPostgres(configuration)
	if err != nil {
		log.Fatalf("Failed to connect to database: %v", err)
	}
	defer dbConnector.Close()
	log.Println("Database connected successfully")

	// Initialize repository
	sessionRepo := repository.NewSessionRepository(dbConnector)

	// Initialize services
	sessionService := service.NewSessionService(configuration, sessionRepo)

	// Initialize aggregator
	aggConfig := &aggregator.Config{
		BatchSize:           configuration.AggBatchSize,
		MaxFramesPerSession: configuration.AggMaxFramesPerSession,
		FlushInterval:       configuration.AggFlushInterval,
		SessionTTL:          configuration.AggSessionTTL,
		WorkerPoolSize:      configuration.AggWorkerPoolSize,
		RequestTimeout:      configuration.AggRequestTimeout,

		// Feature Extraction APIs
		OpenFaceAPIURL:      configuration.OpenFaceAPIURL,
		FrequencyAPIURL:     configuration.FrequencyAPIURL,
		EnableFeatureFusion: configuration.EnableFeatureFusion,

		// Frame Storage
		FrameStorageDir:    configuration.FrameStorageDir,
		FrameBaseURL:       configuration.FrameBaseURL,
		EnableFrameStorage: configuration.EnableFrameStorage,

		// RabbitMQ
		RabbitMQURL:        configuration.RabbitMQURL,
		RabbitMQExchange:   configuration.RabbitMQExchange,
		RabbitMQQueue:      configuration.RabbitMQQueue,
		RabbitMQRoutingKey: configuration.RabbitMQRoutingKey,
		RabbitMQEnabled:    configuration.RabbitMQEnabled,
	}
	frameAggregator := aggregator.NewAggregator(aggConfig)
	frameAggregator.Start()
	log.Println("Frame aggregator initialized and started")

	// Setup HTTP server
	handler := api.NewHandler(sessionService, frameAggregator, configuration)
	router := api.SetupRoutes(handler)
	server := api.NewHTTPServer(configuration, router)

	// Start server in goroutine
	go func() {
		log.Printf("Server starting on %s", configuration.ServerAddress)
		if err := server.ListenAndServe(); err != nil {
			log.Fatalf("Server failed: %v", err)
		}
	}()

	// Wait for interrupt signal
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Println("Shutting down server...")
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Shutdown aggregator first
	frameAggregator.Stop()

	sessionService.Shutdown()
	if err := server.Shutdown(ctx); err != nil {
		log.Fatalf("Server forced to shutdown: %v", err)
	}

	log.Println("Server exited gracefully")
}
