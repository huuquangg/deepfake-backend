```md
## Implementation and Constraints (Go In-Memory Aggregator → FastAPI `/extract/batch`)

### Implementation (Step-by-Step)

1. **Expose an ingestion endpoint in Go**
   - Create `POST /ingest/frame` that accepts `multipart/form-data`:
     - `frame` (single image file, e.g., JPEG)
     - `session_id` (string)
     - optional: `frame_idx` or timestamp for ordering/debug
   - The handler must be **non-blocking**: read the uploaded file into memory, push it to the aggregator, return `200 OK` immediately.

2. **Maintain an in-memory buffer per session**
   - Use a thread-safe map:
     - `map[session_id]*SessionState`
   - `SessionState` contains:
     - `Frames []Frame` (each `Frame` stores `[]byte` + filename + timestamp)
     - `LastSeen time.Time` (last frame arrival)
     - `LastFlushed time.Time` (last batch dispatch time)

3. **Append frames and enforce a sliding window**
   - On each incoming frame:
     - Append to `SessionState.Frames`
     - Update `LastSeen`
     - If `len(Frames) > MaxFramesPerSession` (e.g., 60), drop older frames and keep the newest (sliding window) to prevent backlog and reduce latency.

4. **Flush logic: frame-count trigger**
   - If `len(Frames) >= BatchSize` (BatchSize = 30):
     - Extract a batch of frames (either FIFO first 30, or newest 30 depending on design)
     - Remove them from the session buffer
     - Dispatch the batch asynchronously (goroutine or worker pool)

5. **Flush logic: time-based trigger (timeout)**
   - Run a background ticker (e.g., every 100–250 ms) that checks each active session:
     - If `now - LastFlushed >= FlushInterval` (e.g., 1 second) and `len(Frames) > 0`:
       - Flush up to 30 frames (or all current frames)
       - Clear flushed frames from the buffer
       - Update `LastFlushed`
       - Dispatch asynchronously
   - This keeps latency stable even when frames are dropped or network jitter occurs.

6. **Batch forwarding to FastAPI (multipart upload)**
   - For each batch, build `multipart/form-data`:
     - Add `session_id` as a form field
     - Add `cleanup=true` (or configurable)
     - Add repeated `files` parts (one per frame), matching FastAPI:
       - `files: List[UploadFile] = File(...)`
   - Send `POST http://<fastapi-host>:8001/extract/batch` using a reused `http.Client` (keep-alive) and a strict timeout (e.g., 2–5 seconds).
   - Do not block ingestion while waiting for FastAPI; use async dispatch.

7. **Concurrency control (recommended)**
   - Instead of unbounded goroutines, use a bounded **worker pool**:
     - `jobs chan BatchJob`
     - `N` workers read jobs and call FastAPI
   - This prevents resource spikes under high concurrency.

8. **Session cleanup (TTL)**
   - In the same background loop (or a separate cleanup loop):
     - If `now - LastSeen > SessionTTL` (e.g., 3–5 seconds), delete the session and free memory.
   - This prevents memory leaks when a user disconnects or stops streaming.

---

### Constraints and Guarantees

- **API compatibility constraint**
  - FastAPI `/extract/batch` only accepts `multipart/form-data` with `files: List[UploadFile]`.
  - Therefore, the aggregator must always forward frames as repeated `files` fields (not a single video stream).

- **Latency constraint**
  - The system targets near real-time behavior; buffering must be bounded to ≈1 second.
  - When extraction is slower than ingestion, the design must prioritize freshness (drop old frames) over accumulating delay.

- **Memory constraint (bounded memory)**
  - Memory usage must be capped by:
    - `MaxFramesPerSession` (e.g., 60)
    - `MaxSessions` (configurable upper bound)
    - `SessionTTL` cleanup
  - This prevents unbounded growth over time and under abnormal client behavior.

- **Backpressure constraint**
  - If the FastAPI service is slow/unavailable, the aggregator must not indefinitely queue batches.
  - Use a bounded job queue; if full, drop or degrade (e.g., keep newest frames, discard backlog).

- **Ordering constraint (optional)**
  - If strict frame ordering is required, the client must include `frame_idx`, and the aggregator should sort or enforce monotonicity before batching.
  - If not required, frames can be batched in arrival order for lower overhead.

- **Reliability constraint**
  - The ingestion path should remain responsive even if extraction fails.
  - Failures in batch forwarding should not block `/ingest/frame`; error handling should be isolated to the dispatch layer.

- **Security/abuse constraint**
  - Enforce upload size limits per frame and rate limits per session to mitigate overload or abuse.
```
