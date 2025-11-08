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

	// Video Processing Config
	FrameRate       int
	JPEGQuality     int
	CleanupWindow   time.Duration
	CleanupInterval time.Duration

	// PostgreSQL Config
	PostgresHost     string
	PostgresPort     int
	PostgresUser     string
	PostgresPassword string
	PostgresDB       string
	PostgresSSLMode  string
}

func New() *Config {
	return &Config{
		// Server
		ServerAddress: getEnv("SERVER_ADDRESS", ":8080"),
		TmpDir:        getEnv("TMP_DIR", "tmp"),
		ReadTimeout:   getEnvAsDuration("READ_TIMEOUT", 30*time.Second),
		WriteTimeout:  getEnvAsDuration("WRITE_TIMEOUT", 30*time.Second),
		MaxUploadSize: getEnvAsInt64("MAX_UPLOAD_SIZE", 500*1024*1024), // 500MB

		// Video Processing
		FrameRate:       getEnvAsInt("FRAME_RATE", 30),
		JPEGQuality:     getEnvAsInt("JPEG_QUALITY", 10),
		CleanupWindow:   getEnvAsDuration("CLEANUP_WINDOW", 30*time.Second),
		CleanupInterval: getEnvAsDuration("CLEANUP_INTERVAL", 5*time.Second),

		// PostgreSQL
		PostgresHost:     getEnv("POSTGRES_HOST", "localhost"),
		PostgresPort:     getEnvAsInt("POSTGRES_PORT", 5432),
		PostgresUser:     getEnv("POSTGRES_USER", "postgres"),
		PostgresPassword: getEnv("POSTGRES_PASSWORD", "P@ssw0rd123"),
		PostgresDB:       getEnv("POSTGRES_DB", "streaming_db"),
		PostgresSSLMode:  getEnv("POSTGRES_SSL_MODE", "disable"),
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
