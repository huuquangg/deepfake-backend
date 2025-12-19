package config

import (
	"os"
	"strconv"
	"time"
)

type Config struct {
	// Server Config
	ServerAddress string
	ReadTimeout   time.Duration
	WriteTimeout  time.Duration

	// PostgreSQL Config
	PostgresHost     string
	PostgresPort     int
	PostgresUser     string
	PostgresPassword string
	PostgresDB       string
	PostgresSchema   string
	PostgresSSLMode  string
}

func New() *Config {
	return &Config{
		// Server
		ServerAddress: getEnv("SERVER_ADDRESS", ":8090"),
		ReadTimeout:   getEnvAsDuration("READ_TIMEOUT", 30*time.Second),
		WriteTimeout:  getEnvAsDuration("WRITE_TIMEOUT", 30*time.Second),

		// PostgreSQL
		PostgresHost:     getEnv("POSTGRES_HOST", "localhost"),
		PostgresPort:     getEnvAsInt("POSTGRES_PORT", 5432),
		PostgresUser:     getEnv("POSTGRES_USER", "postgres"),
		PostgresPassword: getEnv("POSTGRES_PASSWORD", "P@ssw0rd123"),
		PostgresDB:       getEnv("POSTGRES_DB", "postgres"),
		PostgresSchema:   getEnv("POSTGRES_SCHEMA", "core_banking"),
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

func getEnvAsDuration(key string, defaultValue time.Duration) time.Duration {
	if value := os.Getenv(key); value != "" {
		if duration, err := time.ParseDuration(value); err == nil {
			return duration
		}
	}
	return defaultValue
}
