package config

import (
	"os"
	"strconv"
	"time"
)

type Config struct {
	// Server Config
	ServerAddress string
	TmpDir        string
	ReadTimeout   time.Duration
	WriteTimeout  time.Duration
	MaxUploadSize int64

	// PostgreSQL Config
	PostgresHost     string
	PostgresPort     int
	PostgresUser     string
	PostgresPassword string
	PostgresDB       string
	PostgresSchema   string
	PostgresSSLMode  string

	// Aggregator Config
	AggBatchSize           int           // Frames per batch (default: 30)
	AggMaxFramesPerSession int           // Max frames buffered per session (default: 60)
	AggFlushInterval       time.Duration // Time-based flush interval (default: 1s)
	AggSessionTTL          time.Duration // Session cleanup timeout (default: 5s)
	AggWorkerPoolSize      int           // Number of workers (default: 4)
	AggRequestTimeout      time.Duration // HTTP request timeout (default: 5s)

	// Feature Extraction APIs
	OpenFaceAPIURL string // OpenFace API URL
	// DeepfakeAPIURL       string // Deepfake detection API URL (uncomment when API available)
	EnableFeatureFusion bool // Enable multi-feature fusion

	// RabbitMQ Config
	RabbitMQURL          string // RabbitMQ connection URL
	RabbitMQExchange     string // Exchange name
	RabbitMQQueue        string // Queue name
	RabbitMQRoutingKey   string // Routing key
	RabbitMQEnabled      bool   // Enable RabbitMQ publishing
}

func New() *Config {
	return &Config{
		// Server
		ServerAddress: getEnv("SERVER_ADDRESS", ":8091"),
		TmpDir:        getEnv("TMP_DIR", "tmp"),
		ReadTimeout:   getEnvAsDuration("READ_TIMEOUT", 30*time.Second),
		WriteTimeout:  getEnvAsDuration("WRITE_TIMEOUT", 30*time.Second),
		MaxUploadSize: getEnvAsInt64("MAX_UPLOAD_SIZE", 500*1024*1024), // 500MB

		// PostgreSQL
		PostgresHost:     getEnv("POSTGRES_HOST", "localhost"),
		PostgresPort:     getEnvAsInt("POSTGRES_PORT", 5432),
		PostgresUser:     getEnv("POSTGRES_USER", "postgres"),
		PostgresPassword: getEnv("POSTGRES_PASSWORD", "P@ssw0rd123"),
		PostgresDB:       getEnv("POSTGRES_DB", "postgres"),
		PostgresSchema:   getEnv("POSTGRES_SCHEMA", "video_streaming"),
		PostgresSSLMode:  getEnv("POSTGRES_SSL_MODE", "disable"),

		// Aggregator
		AggBatchSize:           getEnvAsInt("AGG_BATCH_SIZE", 30),
		AggMaxFramesPerSession: getEnvAsInt("AGG_MAX_FRAMES_PER_SESSION", 60),
		AggFlushInterval:       getEnvAsDuration("AGG_FLUSH_INTERVAL", 1*time.Second),
		AggSessionTTL:          getEnvAsDuration("AGG_SESSION_TTL", 5*time.Second),
		AggWorkerPoolSize:      getEnvAsInt("AGG_WORKER_POOL_SIZE", 4),
		AggRequestTimeout:      getEnvAsDuration("AGG_REQUEST_TIMEOUT", 5*time.Second),

		// Feature Extraction APIs
		OpenFaceAPIURL: getEnv("OPENFACE_API_URL", "http://localhost:8001/extract/batch"),
		// DeepfakeAPIURL:      getEnv("DEEPFAKE_API_URL", "http://localhost:8002/detect/batch"), // Uncomment when API available
		EnableFeatureFusion: getEnvAsBool("ENABLE_FEATURE_FUSION", true),

		// RabbitMQ
		RabbitMQURL:        getEnv("RABBITMQ_URL", "amqp://admin:P@ssw0rd123@localhost:5672/"),
		RabbitMQExchange:   getEnv("RABBITMQ_EXCHANGE", "deepfake.features"),
		RabbitMQQueue:      getEnv("RABBITMQ_QUEUE", "feature.extraction.results"),
		RabbitMQRoutingKey: getEnv("RABBITMQ_ROUTING_KEY", "features.fused"),
		RabbitMQEnabled:    getEnvAsBool("RABBITMQ_ENABLED", true),
	}
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getEnvAsInt(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		if intValue, err := strconv.Atoi(value); err == nil {
			return intValue
		}
	}
	return defaultValue
}

func getEnvAsInt64(key string, defaultValue int64) int64 {
	if value := os.Getenv(key); value != "" {
		if intValue, err := strconv.ParseInt(value, 10, 64); err == nil {
			return intValue
		}
	}
	return defaultValue
}

func getEnvAsDuration(key string, defaultValue time.Duration) time.Duration {
	if value := os.Getenv(key); value != "" {
		if duration, err := time.ParseDuration(value); err == nil {
			return duration
		}
	}
	return defaultValue
}

func getEnvAsBool(key string, defaultValue bool) bool {
	if value := os.Getenv(key); value != "" {
		if boolValue, err := strconv.ParseBool(value); err == nil {
			return boolValue
		}
	}
	return defaultValue
}
