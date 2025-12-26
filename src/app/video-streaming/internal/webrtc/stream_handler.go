package webrtc

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/jpeg"
	"log"
	"sync"
	"time"

	"github.com/pion/webrtc/v3"
	"video-streaming/internal/aggregator"
)

// StreamHandler manages WebRTC connections for real-time frame streaming
type StreamHandler struct {
	aggregator     *aggregator.Aggregator
	peerConnections sync.Map // map[sessionID]*webrtc.PeerConnection
	dataChannels    sync.Map // map[sessionID]*webrtc.DataChannel for frame ingestion
	api            *webrtc.API
	config         webrtc.Configuration
	
	// Frame extraction settings
	frameInterval  time.Duration // Extract frame every N milliseconds (e.g., 66ms for ~15fps)
	jpegQuality    int          // JPEG quality (1-100)
	maxFrameSize   int          // Max frame size in bytes
}

// SessionOffer represents a WebRTC offer from client
type SessionOffer struct {
	SessionID string `json:"session_id"`
	SDP       string `json:"sdp"`
	Type      string `json:"type"`
}

// SessionAnswer represents a WebRTC answer to client
type SessionAnswer struct {
	SessionID string `json:"session_id"`
	SDP       string `json:"sdp"`
	Type      string `json:"type"`
}

// ICECandidate represents an ICE candidate
type ICECandidate struct {
	SessionID string                  `json:"session_id"`
	Candidate webrtc.ICECandidateInit `json:"candidate"`
}

// NewStreamHandler creates a new WebRTC stream handler
func NewStreamHandler(agg *aggregator.Aggregator) *StreamHandler {
	// Configure WebRTC settings
	config := webrtc.Configuration{
		ICEServers: []webrtc.ICEServer{
			{
				URLs: []string{"stun:stun.l.google.com:19302"},
			},
		},
	}

	// Create MediaEngine
	mediaEngine := &webrtc.MediaEngine{}
	if err := mediaEngine.RegisterDefaultCodecs(); err != nil {
		log.Printf("Failed to register default codecs: %v", err)
	}

	// Create API with MediaEngine
	api := webrtc.NewAPI(webrtc.WithMediaEngine(mediaEngine))

	return &StreamHandler{
		aggregator:    agg,
		api:           api,
		config:        config,
		frameInterval: 66 * time.Millisecond, // ~15fps
		jpegQuality:   75,
		maxFrameSize:  2 * 1024 * 1024, // 2MB
	}
}

// HandleOffer processes a WebRTC offer and returns an answer
func (h *StreamHandler) HandleOffer(sessionID string, sdp string) (string, error) {
	// Create new peer connection
	peerConnection, err := h.api.NewPeerConnection(h.config)
	if err != nil {
		return "", fmt.Errorf("failed to create peer connection: %w", err)
	}

	// Store peer connection
	h.peerConnections.Store(sessionID, peerConnection)

	// Setup connection state handlers
	peerConnection.OnConnectionStateChange(func(state webrtc.PeerConnectionState) {
		log.Printf("WebRTC connection state changed: %s (session: %s)", state.String(), sessionID)
		
		if state == webrtc.PeerConnectionStateFailed || 
		   state == webrtc.PeerConnectionStateClosed || 
		   state == webrtc.PeerConnectionStateDisconnected {
			h.CloseSession(sessionID)
		}
	})

	// Setup ICE connection state handler
	peerConnection.OnICEConnectionStateChange(func(state webrtc.ICEConnectionState) {
		log.Printf("ICE connection state changed: %s (session: %s)", state.String(), sessionID)
	})

	// Handle incoming video track (for future full video decoding support)
	peerConnection.OnTrack(func(track *webrtc.TrackRemote, receiver *webrtc.RTPReceiver) {
		log.Printf("Received track: kind=%s, id=%s, streamID=%s (session: %s)", 
			track.Kind(), track.ID(), track.StreamID(), sessionID)
		
		// For mobile integration, frames should come via data channel (see OnDataChannel below)
		// Video track is kept for monitoring/debugging
		if track.Kind() == webrtc.RTPCodecTypeVideo {
			log.Printf("Video track available but using data channel for frame ingestion (session: %s)", sessionID)
		}
	})

	// Create data channel for bidirectional communication
	// Mobile clients will send JPEG frames through "frames" channel
	// Server sends predictions back through "predictions" channel
	predictionsChannel, err := peerConnection.CreateDataChannel("predictions", nil)
	if err != nil {
		log.Printf("Failed to create predictions data channel: %v", err)
	} else {
		h.setupPredictionsChannel(sessionID, predictionsChannel)
	}

	// Handle incoming data channels from client (frames channel)
	peerConnection.OnDataChannel(func(dataChannel *webrtc.DataChannel) {
		log.Printf("Data channel opened by client: label=%s (session: %s)", dataChannel.Label(), sessionID)
		
		if dataChannel.Label() == "frames" {
			h.setupFramesChannel(sessionID, dataChannel)
		}
	})

	// Set remote description (offer from client)
	offer := webrtc.SessionDescription{
		Type: webrtc.SDPTypeOffer,
		SDP:  sdp,
	}
	if err := peerConnection.SetRemoteDescription(offer); err != nil {
		return "", fmt.Errorf("failed to set remote description: %w", err)
	}

	// Create answer
	answer, err := peerConnection.CreateAnswer(nil)
	if err != nil {
		return "", fmt.Errorf("failed to create answer: %w", err)
	}

	// Set local description
	if err := peerConnection.SetLocalDescription(answer); err != nil {
		return "", fmt.Errorf("failed to set local description: %w", err)
	}

	// Return answer SDP
	return answer.SDP, nil
}

// HandleICECandidate adds an ICE candidate to the peer connection
func (h *StreamHandler) HandleICECandidate(sessionID string, candidate webrtc.ICECandidateInit) error {
	val, ok := h.peerConnections.Load(sessionID)
	if !ok {
		return fmt.Errorf("peer connection not found for session: %s", sessionID)
	}

	peerConnection := val.(*webrtc.PeerConnection)
	if err := peerConnection.AddICECandidate(candidate); err != nil {
		return fmt.Errorf("failed to add ICE candidate: %w", err)
	}

	return nil
}

// ingestJPEGFrame ingests a JPEG frame to aggregator
func (h *StreamHandler) ingestJPEGFrame(sessionID string, jpegData []byte, frameNum int) {
	// Validate JPEG data
	if len(jpegData) == 0 {
		log.Printf("Received empty frame data (session: %s)", sessionID)
		return
	}
	
	// Check if data is valid JPEG (starts with FF D8)
	if len(jpegData) < 2 || jpegData[0] != 0xFF || jpegData[1] != 0xD8 {
		log.Printf("Invalid JPEG data received (session: %s, size: %d bytes)", sessionID, len(jpegData))
		return
	}
	
	// Generate filename with timestamp and frame number
	filename := fmt.Sprintf("%s_webrtc_%d_%04d.jpeg", sessionID, time.Now().Unix(), frameNum)
	
	// Ingest to aggregator (non-blocking)
	if err := h.aggregator.IngestFrame(sessionID, jpegData, filename); err != nil {
		log.Printf("Failed to ingest WebRTC frame: %v (session: %s, frame: %d)", err, sessionID, frameNum)
	} else {
		if frameNum%30 == 0 { // Log every 30 frames (~2 seconds at 15fps)
			log.Printf("WebRTC frame ingested: session=%s, frame=%d, size=%d bytes", sessionID, frameNum, len(jpegData))
		}
	}
}

// setupFramesChannel handles incoming JPEG frames from mobile client
func (h *StreamHandler) setupFramesChannel(sessionID string, channel *webrtc.DataChannel) {
	// Store channel reference
	h.dataChannels.Store(sessionID+"_frames", channel)
	
	frameCounter := 0
	
	channel.OnOpen(func() {
		log.Printf("Frames data channel opened (session: %s) - ready to receive JPEG frames", sessionID)
	})

	channel.OnClose(func() {
		log.Printf("Frames data channel closed (session: %s)", sessionID)
		h.dataChannels.Delete(sessionID + "_frames")
	})

	// Handle incoming JPEG frame data from mobile client
	channel.OnMessage(func(msg webrtc.DataChannelMessage) {
		if !msg.IsString {
			// Binary data = JPEG frame
			frameCounter++
			h.ingestJPEGFrame(sessionID, msg.Data, frameCounter)
		} else {
			// Text message = control/metadata
			log.Printf("Frames channel control message: %s (session: %s)", string(msg.Data), sessionID)
		}
	})
}

// setupPredictionsChannel handles sending predictions back to mobile client
func (h *StreamHandler) setupPredictionsChannel(sessionID string, channel *webrtc.DataChannel) {
	// Store channel reference for sending predictions
	h.dataChannels.Store(sessionID+"_predictions", channel)
	
	channel.OnOpen(func() {
		log.Printf("Predictions data channel opened (session: %s) - ready to send predictions", sessionID)
	})

	channel.OnClose(func() {
		log.Printf("Predictions data channel closed (session: %s)", sessionID)
		h.dataChannels.Delete(sessionID + "_predictions")
	})

	channel.OnMessage(func(msg webrtc.DataChannelMessage) {
		log.Printf("Predictions channel message: %s (session: %s)", string(msg.Data), sessionID)
	})
}

// SendPrediction sends a prediction result back to mobile client via data channel
func (h *StreamHandler) SendPrediction(sessionID string, prediction map[string]interface{}) error {
	// Get predictions data channel
	val, ok := h.dataChannels.Load(sessionID + "_predictions")
	if !ok {
		return fmt.Errorf("predictions data channel not found for session: %s", sessionID)
	}

	channel := val.(*webrtc.DataChannel)
	
	// Check if channel is open
	if channel.ReadyState() != webrtc.DataChannelStateOpen {
		return fmt.Errorf("predictions data channel not open (state: %s)", channel.ReadyState().String())
	}

	// Serialize prediction to JSON
	data, err := json.Marshal(prediction)
	if err != nil {
		return fmt.Errorf("failed to marshal prediction: %w", err)
	}

	// Send via data channel
	if err := channel.SendText(string(data)); err != nil {
		return fmt.Errorf("failed to send prediction: %w", err)
	}
	
	log.Printf("Prediction sent: session=%s, label=%s, confidence=%.2f", 
		sessionID, prediction["label"], prediction["confidence"])
	
	return nil
}

// CloseSession closes WebRTC connection for a session
func (h *StreamHandler) CloseSession(sessionID string) error {
	val, ok := h.peerConnections.Load(sessionID)
	if !ok {
		return fmt.Errorf("peer connection not found for session: %s", sessionID)
	}

	peerConnection := val.(*webrtc.PeerConnection)
	
	if err := peerConnection.Close(); err != nil {
		log.Printf("Error closing peer connection: %v (session: %s)", err, sessionID)
	}

	// Clean up data channels
	h.dataChannels.Delete(sessionID + "_frames")
	h.dataChannels.Delete(sessionID + "_predictions")
	
	h.peerConnections.Delete(sessionID)
	log.Printf("WebRTC session closed: %s", sessionID)
	
	return nil
}

// GetSessionStats returns statistics for all WebRTC sessions
func (h *StreamHandler) GetSessionStats() map[string]interface{} {
	stats := make(map[string]interface{})
	activeSessions := 0
	
	h.peerConnections.Range(func(key, value interface{}) bool {
		activeSessions++
		sessionID := key.(string)
		peerConnection := value.(*webrtc.PeerConnection)
		
		stats[sessionID] = map[string]interface{}{
			"connection_state": peerConnection.ConnectionState().String(),
			"ice_state":        peerConnection.ICEConnectionState().String(),
		}
		
		return true
	})
	
	stats["total_active_sessions"] = activeSessions
	stats["frame_interval_ms"] = h.frameInterval.Milliseconds()
	stats["target_fps"] = 1000 / h.frameInterval.Milliseconds()
	
	return stats
}

// Helper: Convert image to JPEG bytes
func imageToJPEG(img image.Image, quality int) ([]byte, error) {
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: quality}); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// Helper: Decode base64 image
func decodeBase64Image(b64 string) ([]byte, error) {
	return base64.StdEncoding.DecodeString(b64)
}
