package aggregator

import (
	"bytes"
	"encoding/json"
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"sync"
	"time"

	amqp "github.com/rabbitmq/amqp091-go"
)

// Frame represents a single frame in memory
type Frame struct {
	Data      []byte
	Filename  string
	Timestamp time.Time
}

// SessionState holds the buffered frames for a session
type SessionState struct {
	Frames       []Frame
	LastSeen     time.Time
	LastFlushed  time.Time
	mu           sync.Mutex
}

// BatchJob represents a batch to be forwarded
type BatchJob struct {
	SessionID string
	Frames    []Frame
}

// Config holds aggregator configuration
type Config struct {
	BatchSize           int           // Number of frames to trigger batch (default: 30)
	MaxFramesPerSession int           // Maximum frames to keep per session (default: 60)
	FlushInterval       time.Duration // Time-based flush interval (default: 1s)
	SessionTTL          time.Duration // Session cleanup timeout (default: 5s)
	WorkerPoolSize      int           // Number of workers for batch forwarding (default: 4)
	RequestTimeout      time.Duration // HTTP request timeout (default: 5s)

	// Feature Extraction APIs
	OpenFaceAPIURL string // OpenFace API URL
	// DeepfakeAPIURL      string // Deepfake detection API URL (uncomment when API available)
	EnableFeatureFusion bool // Enable multi-feature fusion

	// RabbitMQ Config
	RabbitMQURL        string // RabbitMQ connection URL
	RabbitMQExchange   string // Exchange name
	RabbitMQQueue      string // Queue name
	RabbitMQRoutingKey string // Routing key
	RabbitMQEnabled    bool   // Enable RabbitMQ publishing
}

// Aggregator manages frame aggregation and batching
type Aggregator struct {
	config     *Config
	sessions   sync.Map // map[sessionID]*SessionState
	jobQueue   chan BatchJob
	client     *http.Client
	stopCh     chan struct{}
	wg         sync.WaitGroup
	rabbitConn *amqp.Connection
	rabbitChan *amqp.Channel
}

// FeatureResult holds results from a single feature extraction API
type FeatureResult struct {
	APIName  string                 `json:"api_name"`
	Features map[string]interface{} `json:"features"`
	Error    string                 `json:"error,omitempty"`
}

// FusedFeatures holds combined results from all APIs
type FusedFeatures struct {
	SessionID    string        `json:"session_id"`
	BatchID      string        `json:"batch_id"`
	Timestamp    time.Time     `json:"timestamp"`
	FrameCount   int           `json:"frame_count"`
	OpenFace     FeatureResult `json:"openface,omitempty"`
	Deepfake     FeatureResult `json:"deepfake,omitempty"`
	Fused        interface{}   `json:"fused,omitempty"`        // Combined/fused features
	ProcessingMS int64         `json:"processing_ms"`
}

// NewAggregator creates a new frame aggregator
func NewAggregator(config *Config) *Aggregator {
	if config.BatchSize == 0 {
		config.BatchSize = 30
	}
	if config.MaxFramesPerSession == 0 {
		config.MaxFramesPerSession = 60
	}
	if config.FlushInterval == 0 {
		config.FlushInterval = 1 * time.Second
	}
	if config.SessionTTL == 0 {
		config.SessionTTL = 5 * time.Second
	}
	if config.WorkerPoolSize == 0 {
		config.WorkerPoolSize = 4
	}
	if config.RequestTimeout == 0 {
		config.RequestTimeout = 5 * time.Second
	}

	agg := &Aggregator{
		config: config,
		jobQueue: make(chan BatchJob, config.WorkerPoolSize*2), // Buffered channel
		client: &http.Client{
			Timeout: config.RequestTimeout,
			Transport: &http.Transport{
				MaxIdleConns:        100,
				MaxIdleConnsPerHost: 10,
				IdleConnTimeout:     90 * time.Second,
			},
		},
		stopCh: make(chan struct{}),
	}

	// Initialize RabbitMQ if enabled
	if config.RabbitMQEnabled {
		if err := agg.initRabbitMQ(); err != nil {
			log.Printf("Warning: Failed to initialize RabbitMQ: %v. Will continue without RabbitMQ.", err)
		}
	}

	return agg
}

// Start initializes the aggregator background processes
func (a *Aggregator) Start() {
	log.Println("Starting aggregator...")

	// Start worker pool
	for i := 0; i < a.config.WorkerPoolSize; i++ {
		a.wg.Add(1)
		go a.worker(i)
	}

	// Start background ticker for time-based flushing and cleanup
	a.wg.Add(1)
	go a.backgroundTicker()

	log.Printf("Aggregator started with %d workers", a.config.WorkerPoolSize)
}

// Stop gracefully shuts down the aggregator
func (a *Aggregator) Stop() {
	log.Println("Stopping aggregator...")
	close(a.stopCh)
	a.wg.Wait()
	close(a.jobQueue)

	// Close RabbitMQ connections
	if a.rabbitChan != nil {
		a.rabbitChan.Close()
	}
	if a.rabbitConn != nil {
		a.rabbitConn.Close()
	}

	log.Println("Aggregator stopped")
}

// initRabbitMQ initializes RabbitMQ connection and channel
func (a *Aggregator) initRabbitMQ() error {
	var err error
	a.rabbitConn, err = amqp.Dial(a.config.RabbitMQURL)
	if err != nil {
		return err
	}

	a.rabbitChan, err = a.rabbitConn.Channel()
	if err != nil {
		return err
	}

	// Declare exchange
	err = a.rabbitChan.ExchangeDeclare(
		a.config.RabbitMQExchange, // name
		"topic",                    // type
		true,                       // durable
		false,                      // auto-deleted
		false,                      // internal
		false,                      // no-wait
		nil,                        // arguments
	)
	if err != nil {
		return err
	}

	// Declare queue
	_, err = a.rabbitChan.QueueDeclare(
		a.config.RabbitMQQueue, // name
		true,                   // durable
		false,                  // delete when unused
		false,                  // exclusive
		false,                  // no-wait
		nil,                    // arguments
	)
	if err != nil {
		return err
	}

	// Bind queue to exchange
	err = a.rabbitChan.QueueBind(
		a.config.RabbitMQQueue,      // queue name
		a.config.RabbitMQRoutingKey, // routing key
		a.config.RabbitMQExchange,   // exchange
		false,
		nil,
	)

	log.Printf("RabbitMQ initialized: exchange=%s, queue=%s, routing_key=%s",
		a.config.RabbitMQExchange, a.config.RabbitMQQueue, a.config.RabbitMQRoutingKey)

	return err
}

// publishToRabbitMQ publishes fused features to RabbitMQ
func (a *Aggregator) publishToRabbitMQ(features *FusedFeatures) error {
	if !a.config.RabbitMQEnabled || a.rabbitChan == nil {
		return nil
	}

	body, err := json.Marshal(features)
	if err != nil {
		return err
	}

	err = a.rabbitChan.Publish(
		a.config.RabbitMQExchange,   // exchange
		a.config.RabbitMQRoutingKey, // routing key
		false,                       // mandatory
		false,                       // immediate
		amqp.Publishing{
			ContentType:  "application/json",
			Body:         body,
			DeliveryMode: amqp.Persistent,
			Timestamp:    time.Now(),
		},
	)

	return err
}

// IngestFrame adds a frame to the session buffer
func (a *Aggregator) IngestFrame(sessionID string, frameData []byte, filename string) error {
	// Get or create session state
	stateInterface, _ := a.sessions.LoadOrStore(sessionID, &SessionState{
		Frames:      make([]Frame, 0, a.config.MaxFramesPerSession),
		LastSeen:    time.Now(),
		LastFlushed: time.Now(),
	})
	state := stateInterface.(*SessionState)

	state.mu.Lock()
	defer state.mu.Unlock()

	// Create frame
	frame := Frame{
		Data:      frameData,
		Filename:  filename,
		Timestamp: time.Now(),
	}

	// Append frame and enforce sliding window
	state.Frames = append(state.Frames, frame)
	state.LastSeen = time.Now()

	// Enforce max frames (keep newest frames)
	if len(state.Frames) > a.config.MaxFramesPerSession {
		// Drop oldest frames
		excess := len(state.Frames) - a.config.MaxFramesPerSession
		state.Frames = state.Frames[excess:]
	}

	// Check if we should flush based on frame count
	if len(state.Frames) >= a.config.BatchSize {
		a.flushSession(sessionID, state, false)
	}

	return nil
}

// flushSession extracts a batch and sends to job queue
// Must be called with state.mu held
func (a *Aggregator) flushSession(sessionID string, state *SessionState, timeBased bool) {
	if len(state.Frames) == 0 {
		return
	}

	// Determine batch size
	batchSize := a.config.BatchSize
	if batchSize > len(state.Frames) {
		batchSize = len(state.Frames)
	}

	// Extract batch (newest frames for freshness)
	batch := make([]Frame, batchSize)
	copy(batch, state.Frames[:batchSize])

	// Remove flushed frames
	state.Frames = state.Frames[batchSize:]
	state.LastFlushed = time.Now()

	// Send to job queue (non-blocking)
	job := BatchJob{
		SessionID: sessionID,
		Frames:    batch,
	}

	select {
	case a.jobQueue <- job:
		flushType := "frame-count"
		if timeBased {
			flushType = "time-based"
		}
		log.Printf("Flushed batch for session %s (%s): %d frames", sessionID, flushType, len(batch))
	default:
		log.Printf("Warning: Job queue full, dropping batch for session %s", sessionID)
	}
}

// backgroundTicker handles time-based flushing and session cleanup
func (a *Aggregator) backgroundTicker() {
	defer a.wg.Done()

	ticker := time.NewTicker(a.config.FlushInterval)
	defer ticker.Stop()

	for {
		select {
		case <-a.stopCh:
			return
		case <-ticker.C:
			a.performTimedFlushAndCleanup()
		}
	}
}

// performTimedFlushAndCleanup checks all sessions for timed flush and TTL cleanup
func (a *Aggregator) performTimedFlushAndCleanup() {
	now := time.Now()

	a.sessions.Range(func(key, value interface{}) bool {
		sessionID := key.(string)
		state := value.(*SessionState)

		state.mu.Lock()
		defer state.mu.Unlock()

		// Check for session TTL cleanup
		if now.Sub(state.LastSeen) > a.config.SessionTTL {
			log.Printf("Cleaning up session %s (TTL exceeded)", sessionID)
			a.sessions.Delete(sessionID)
			return true
		}

		// Check for time-based flush
		if now.Sub(state.LastFlushed) >= a.config.FlushInterval && len(state.Frames) > 0 {
			a.flushSession(sessionID, state, true)
		}

		return true
	})
}

// worker processes batch jobs from the queue
func (a *Aggregator) worker(id int) {
	defer a.wg.Done()
	log.Printf("Worker %d started", id)

	for {
		select {
		case <-a.stopCh:
			log.Printf("Worker %d stopping", id)
			return
		case job, ok := <-a.jobQueue:
			if !ok {
				return
			}
			a.processBatch(id, job)
		}
	}
}

// processBatch forwards a batch to multiple APIs and fuses results
func (a *Aggregator) processBatch(workerID int, job BatchJob) {
	start := time.Now()
	batchID := generateBatchID(job.SessionID)

	fusedResult := &FusedFeatures{
		SessionID:  job.SessionID,
		BatchID:    batchID,
		Timestamp:  start,
		FrameCount: len(job.Frames),
	}

	// Call APIs in parallel if feature fusion is enabled
	if a.config.EnableFeatureFusion {
		var wg sync.WaitGroup
		var mu sync.Mutex

		// Call OpenFace API
		if a.config.OpenFaceAPIURL != "" {
			wg.Add(1)
			go func() {
				defer wg.Done()
				result := a.callFeatureAPI(workerID, "OpenFace", a.config.OpenFaceAPIURL, job)
				mu.Lock()
				fusedResult.OpenFace = result
				mu.Unlock()
			}()
		}

		// Call Deepfake API (uncomment when API is available)
		// if a.config.DeepfakeAPIURL != "" {
		// 	wg.Add(1)
		// 	go func() {
		// 		defer wg.Done()
		// 		result := a.callFeatureAPI(workerID, "Deepfake", a.config.DeepfakeAPIURL, job)
		// 		mu.Lock()
		// 		fusedResult.Deepfake = result
		// 		mu.Unlock()
		// 	}()
		// }

		wg.Wait()

		// Fuse the features after both APIs return
		fusedResult.Fused = a.fuseFeatures(fusedResult.OpenFace, fusedResult.Deepfake)
	} else {
		// fusedResult.OpenFace = a.callFeatureAPI(workerID, "OpenFace", a.config.OpenFaceAPIURL, job)
	}

	fusedResult.ProcessingMS = time.Since(start).Milliseconds()

	// Publish to RabbitMQ
	if a.config.RabbitMQEnabled {
		if err := a.publishToRabbitMQ(fusedResult); err != nil {
			log.Printf("Worker %d: Failed to publish to RabbitMQ for session %s: %v", workerID, job.SessionID, err)
		} else {
			log.Printf("Worker %d: Published fused features to RabbitMQ for session %s (batch %s)", workerID, job.SessionID, batchID)
		}
	}

	log.Printf("Worker %d: Processed batch for session %s (%d frames, %d APIs) in %v",
		workerID, job.SessionID, len(job.Frames), countAPIs(fusedResult), time.Since(start))
}

// callFeatureAPI calls a single feature extraction API
func (a *Aggregator) callFeatureAPI(workerID int, apiName, apiURL string, job BatchJob) FeatureResult {
	result := FeatureResult{
		APIName:  apiName,
		Features: make(map[string]interface{}),
	}

	// Build multipart form
	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)

	// Add session_id field
	if err := writer.WriteField("session_id", job.SessionID); err != nil {
		result.Error = err.Error()
		return result
	}

	// Add cleanup field
	if err := writer.WriteField("cleanup", "true"); err != nil {
		result.Error = err.Error()
		return result
	}

	// Add each frame as a file part
	for i, frame := range job.Frames {
		part, err := writer.CreateFormFile("files", frame.Filename)
		if err != nil {
			result.Error = err.Error()
			return result
		}
		if _, err := io.Copy(part, bytes.NewReader(frame.Data)); err != nil {
			result.Error = err.Error()
			return result
		}
		_ = i // avoid unused variable
	}

	if err := writer.Close(); err != nil {
		result.Error = err.Error()
		return result
	}

	// Create HTTP request
	req, err := http.NewRequest("POST", apiURL, body)
	if err != nil {
		result.Error = err.Error()
		return result
	}
	req.Header.Set("Content-Type", writer.FormDataContentType())

	// Send request
	resp, err := a.client.Do(req)
	if err != nil {
		result.Error = err.Error()
		log.Printf("Worker %d: %s API error for session %s: %v", workerID, apiName, job.SessionID, err)
		return result
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		result.Error = string(bodyBytes)
		log.Printf("Worker %d: %s API returned status %d for session %s: %s", workerID, apiName, resp.StatusCode, job.SessionID, result.Error)
		return result
	}

	// Parse response
	if err := json.NewDecoder(resp.Body).Decode(&result.Features); err != nil {
		result.Error = err.Error()
		return result
	}

	log.Printf("Worker %d: %s API successful for session %s (%d frames)", workerID, apiName, job.SessionID, len(job.Frames))
	return result
}

// generateBatchID creates a unique batch identifier
func generateBatchID(sessionID string) string {
	return sessionID + "_" + time.Now().Format("20060102_150405_000")
}

// countAPIs counts how many APIs returned results
func countAPIs(fused *FusedFeatures) int {
	count := 0
	if fused.OpenFace.APIName != "" {
		count++
	}
	if fused.Deepfake.APIName != "" {
		count++
	}
	return count
}

// fuseFeatures combines features from multiple APIs
// This is where you implement your fusion logic
func (a *Aggregator) fuseFeatures(openface, deepfake FeatureResult) interface{} {
	fused := make(map[string]interface{})

	// Combine OpenFace features
	if openface.Error == "" && len(openface.Features) > 0 {
		fused["openface_features"] = openface.Features
		fused["openface_available"] = true
	} else {
		fused["openface_available"] = false
		if openface.Error != "" {
			fused["openface_error"] = openface.Error
		}
	}

	// Combine Deepfake features
	if deepfake.Error == "" && len(deepfake.Features) > 0 {
		fused["deepfake_features"] = deepfake.Features
		fused["deepfake_available"] = true
	} else {
		fused["deepfake_available"] = false
		if deepfake.Error != "" {
			fused["deepfake_error"] = deepfake.Error
		}
	}

	// Add fusion metadata
	fused["fusion_strategy"] = "concatenation" // or "weighted_average", "attention", etc.
	fused["fusion_timestamp"] = time.Now()

	// TODO: Implement your custom fusion logic here
	// Examples:
	// - Concatenate feature vectors
	// - Weighted averaging
	// - Attention-based fusion
	// - Neural network-based fusion
	// - Statistical fusion (mean, max, etc.)

	return fused
}

// GetSessionStats returns current session statistics
func (a *Aggregator) GetSessionStats() map[string]interface{} {
	stats := map[string]interface{}{
		"active_sessions": 0,
		"total_frames":    0,
		"queue_length":    len(a.jobQueue),
	}

	sessionCount := 0
	totalFrames := 0

	a.sessions.Range(func(key, value interface{}) bool {
		sessionCount++
		state := value.(*SessionState)
		state.mu.Lock()
		totalFrames += len(state.Frames)
		state.mu.Unlock()
		return true
	})

	stats["active_sessions"] = sessionCount
	stats["total_frames"] = totalFrames

	return stats
}
