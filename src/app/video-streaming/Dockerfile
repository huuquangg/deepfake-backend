FROM golang:1.24-alpine AS builder

# Install FFmpeg
RUN apk add --no-cache ffmpeg

WORKDIR /app

# Copy go mod files
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Build the application
RUN CGO_ENABLED=0 GOOS=linux go build -o /app/bin/server ./cmd/server

# Final stage
FROM alpine:latest

# Install FFmpeg in final image
RUN apk add --no-cache ffmpeg

WORKDIR /app

# Copy binary from builder
COPY --from=builder /app/bin/server /app/server

# Create tmp directory
RUN mkdir -p /app/tmp

# Expose port
EXPOSE 8080

# Run the application
CMD ["/app/server"]