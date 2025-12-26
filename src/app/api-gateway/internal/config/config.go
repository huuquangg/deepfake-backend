package config

import (
	"log"
	"net/url"
	"os"
	"strings"
)

type Config struct {
	ListenAddr           string
	VideoStreamingURL    *url.URL
	VideoStreamingSocket *url.URL
	OpenFaceURL          *url.URL
	OpenFaceBatchURL     *url.URL
	FrequencyURL         *url.URL
	CoreBankingURL       *url.URL
}

func Load() *Config {
	return &Config{
		ListenAddr:           getEnv("GATEWAY_ADDR", ":8096"),
		VideoStreamingURL:    mustParseURL(getEnv("VIDEO_STREAMING_URL", "http://localhost:8091")),
		VideoStreamingSocket: mustParseURL(getEnv("VIDEO_STREAMING_SOCKET_URL", "http://localhost:8093")),
		OpenFaceURL:          mustParseURL(getEnv("OPENFACE_URL", "http://localhost:8000")),
		OpenFaceBatchURL:     mustParseURL(getEnv("OPENFACE_BATCH_URL", "http://localhost:8001")),
		FrequencyURL:         mustParseURL(getEnv("FREQUENCY_URL", "http://localhost:8092")),
		CoreBankingURL:       mustParseURL(getEnv("CORE_BANKING_URL", "http://localhost:8090")),
	}
}

func getEnv(key, fallback string) string {
	value := strings.TrimSpace(os.Getenv(key))
	if value == "" {
		return fallback
	}
	return value
}

func mustParseURL(raw string) *url.URL {
	parsed, err := url.Parse(raw)
	if err != nil {
		log.Fatalf("invalid url %q: %v", raw, err)
	}
	if parsed.Scheme == "" || parsed.Host == "" {
		log.Fatalf("invalid url %q: missing scheme or host", raw)
	}
	return parsed
}
