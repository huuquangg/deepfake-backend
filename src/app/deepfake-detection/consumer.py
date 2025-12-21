#!/usr/bin/env python3
"""
Deepfake Detection Consumer

Key behaviors:
- Consumes fused per-frame features + frame URIs from RabbitMQ.
- Builds fixed-length model inputs: images (1, 15, 224, 224, 3) and CSV features (1, 15, 957).
- Padding: if a frame/feature is missing, repeats the last available item; if none exist yet, pads with zeros.
- Logs every message receipt and every ACK/NACK (with delivery_tag, session_id, batch_id, and reason).
- Graceful shutdown on SIGINT/SIGTERM (stops consuming and closes RabbitMQ connection).
- Optional deletion: if DELETE_FRAMES=1, attempts HTTP DELETE on each frame URI after successful ACK.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import asyncio
import threading

import cv2
import joblib
import numpy as np
import pika
import requests
import socketio
from aiohttp import web
from tensorflow import keras

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("deepfake-detection-consumer")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip().lower()
    return raw in {"1", "true", "t", "yes", "y", "on"}


@dataclass(frozen=True)
class Config:
    # RabbitMQ
    rabbitmq_url: str = os.getenv("RABBITMQ_URL", "amqp://admin:P@ssw0rd123@localhost:5672/")
    rabbitmq_queue: str = os.getenv("RABBITMQ_QUEUE", "feature.extraction.results")

    # Model
    model_path: str = os.getenv(
        "MODEL_PATH",
        "/home/huuquangdang/huu.quang.dang/thesis/deepfake-1801/src/app/deepfake-detection/TCN_TemporalConvNet_Residual Blocks_final.h5",
    )
    scaler_path: str = os.getenv(
        "SCALER_PATH",
        "/home/huuquangdang/huu.quang.dang/thesis/deepfake-1801/src/app/deepfake-detection/csv_scaler.pkl",
    )

    # Socket.IO Server (Option B: Backend is server, mobile is client)
    socketio_enabled: bool = _env_bool("SOCKETIO_ENABLED", True)
    socketio_host: str = os.getenv("SOCKETIO_HOST", "0.0.0.0")
    socketio_port: int = int(os.getenv("SOCKETIO_PORT", "8093"))
    socketio_event: str = os.getenv("SOCKETIO_EVENT", "prediction")
    socketio_use_rooms: bool = _env_bool("SOCKETIO_USE_ROOMS", True)  # Session-based room filtering

    # Input specs
    sequence_length: int = int(os.getenv("SEQUENCE_LENGTH", "15"))
    image_size: Tuple[int, int] = (224, 224)
    csv_features: int = 952  # 283 frequency + 669 openface (excluding 5 metadata cols)

    # Frame fetching
    frame_fetch_timeout_s: float = float(os.getenv("FRAME_FETCH_TIMEOUT", "5"))
    frame_cache_size: int = int(os.getenv("FRAME_CACHE_SIZE", "100"))

    # Processing behavior
    pad_sequence: bool = _env_bool("PAD_SEQUENCE", True)  # WARNING: Padding short sequences (<15 frames) may degrade accuracy
    window_start_mode: str = os.getenv("WINDOW_START_MODE", "min")  # "min" | "zero"
    requeue_on_error: bool = _env_bool("REQUEUE_ON_ERROR", False)

    # Optional deletion
    delete_frames: bool = _env_bool("DELETE_FRAMES", False)
    delete_timeout_s: float = float(os.getenv("DELETE_TIMEOUT", "2"))


def of_fixed_columns() -> List[str]:
    """
    Define the 669 OpenFace feature columns (excluding metadata)
    Original has 674 columns, but we exclude: frame, face_id, timestamp, confidence, success
    
    Returns:
        List of column names for the 669 features (no metadata)
    """
    cols: List[str] = []
    # Skip: frame, face_id, timestamp, confidence, success (5 metadata columns)
    cols += ["gaze_0_x","gaze_0_y","gaze_0_z","gaze_1_x","gaze_1_y","gaze_1_z","gaze_angle_x","gaze_angle_y"]
    cols += [f"eye_lmk_x_{i}" for i in range(56)]
    cols += [f"eye_lmk_y_{i}" for i in range(56)]
    cols += [f"eye_lmk_X_{i}" for i in range(56)]
    cols += [f"eye_lmk_Y_{i}" for i in range(56)]
    cols += [f"eye_lmk_Z_{i}" for i in range(56)]
    cols += ["pose_Tx","pose_Ty","pose_Tz","pose_Rx","pose_Ry","pose_Rz"]
    cols += [f"x_{i}" for i in range(68)]
    cols += [f"y_{i}" for i in range(68)]
    cols += [f"X_{i}" for i in range(68)]
    cols += [f"Y_{i}" for i in range(68)]
    cols += [f"Z_{i}" for i in range(68)]
    au_r = ["AU01","AU02","AU04","AU05","AU06","AU07","AU09","AU10","AU12","AU14","AU15","AU17","AU20","AU23","AU25","AU26","AU45"]
    au_c = ["AU01","AU02","AU04","AU05","AU06","AU07","AU09","AU10","AU12","AU14","AU15","AU17","AU20","AU23","AU25","AU26","AU28","AU45"]
    cols += [f"{a}_r" for a in au_r]
    cols += [f"{a}_c" for a in au_c]
    return cols


class FrameFetcher:
    def __init__(self, cache_size: int, timeout_s: float, image_size: Tuple[int, int]):
        self._cache: Dict[str, Tuple[np.ndarray, float]] = {}
        self._cache_size = cache_size
        self._timeout_s = timeout_s
        self._image_size = image_size
        self._session = requests.Session()

    def fetch_frame(self, uri: str) -> Optional[np.ndarray]:
        if not uri:
            return None

        cached = self._cache.get(uri)
        if cached is not None:
            return cached[0]

        try:
            resp = self._session.get(uri, timeout=self._timeout_s)
            resp.raise_for_status()

            img_array = np.frombuffer(resp.content, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                logger.warning("frame_decode_failed uri=%s", uri)
                return None

            img = cv2.resize(img, self._image_size)
            img = img.astype(np.float32) / 255.0  # keep BGR

            self._cache[uri] = (img, time.time())
            if len(self._cache) > self._cache_size:
                oldest_uri = min(self._cache.keys(), key=lambda k: self._cache[k][1])
                self._cache.pop(oldest_uri, None)

            return img
        except Exception as e:
            logger.warning("frame_fetch_failed uri=%s err=%s", uri, e)
            return None

    def delete_frame(self, uri: str, timeout_s: float) -> bool:
        if not uri:
            return False
        try:
            resp = self._session.delete(uri, timeout=timeout_s)
            if 200 <= resp.status_code < 300 or resp.status_code == 404:
                return True
            logger.warning("frame_delete_failed uri=%s status=%s", uri, resp.status_code)
            return False
        except Exception as e:
            logger.warning("frame_delete_error uri=%s err=%s", uri, e)
            return False

    def delete_batch(self, uris: List[str], timeout_s: float) -> None:
        for uri in uris:
            self.delete_frame(uri, timeout_s=timeout_s)


class InputBuilder:
    def __init__(self, config: Config, frame_fetcher: FrameFetcher):
        self._cfg = config
        self._frame_fetcher = frame_fetcher
        self._of_columns = of_fixed_columns()

        if os.path.exists(self._cfg.scaler_path):
            self._scaler = joblib.load(self._cfg.scaler_path)
            logger.info("scaler_loaded path=%s", self._cfg.scaler_path)
        else:
            self._scaler = None
            logger.warning("scaler_missing path=%s (scaling disabled)", self._cfg.scaler_path)

    @staticmethod
    def _safe_float(v: Any) -> float:
        if v is None:
            return 0.0
        if isinstance(v, (int, float, np.number)):
            return float(v)
        try:
            return float(str(v))
        except Exception:
            return 0.0

    def _extract_frequency_features_283(self, row: Dict[str, Any]) -> np.ndarray:
        vals: List[float] = []

        # SRM: 20 kernels × 6 stats = 120, ordered per kernel
        for i in range(1, 21):
            vals.extend(
                [
                    self._safe_float(row.get(f"SRM_mean_{i}")),
                    self._safe_float(row.get(f"SRM_var_{i}")),
                    self._safe_float(row.get(f"SRM_skew_{i}")),
                    self._safe_float(row.get(f"SRM_kurt_{i}")),
                    self._safe_float(row.get(f"SRM_entropy_{i}")),
                    self._safe_float(row.get(f"SRM_energy_{i}")),
                ]
            )

        # DCT: 4 stats × 3 bands + 3 entropies + 1 energy + 20 zigzag + 24 hist = 60
        for band in ("low", "mid", "high"):
            vals.extend(
                [
                    self._safe_float(row.get(f"DCT_mean_{band}")),
                    self._safe_float(row.get(f"DCT_var_{band}")),
                    self._safe_float(row.get(f"DCT_skew_{band}")),
                    self._safe_float(row.get(f"DCT_kurt_{band}")),
                ]
            )
        for band in ("low", "mid", "high"):
            vals.append(self._safe_float(row.get(f"DCT_entropy_{band}")))
        vals.append(self._safe_float(row.get("DCT_energy_total")))
        for i in range(20):
            vals.append(self._safe_float(row.get(f"DCT_zigzag_{i}")))
        for band in ("low", "mid", "high"):
            for bin_idx in range(8):
                vals.append(self._safe_float(row.get(f"DCT_hist_{band}_bin_{bin_idx}")))

        # FFT globals: 13
        fft_globals = [
            "fft_psd_total",
            "fft_E_low",
            "fft_E_mid",
            "fft_E_high",
            "fft_E_high_over_low",
            "fft_E_mid_over_low",
            "fft_radial_centroid",
            "fft_radial_bandwidth",
            "fft_rolloff_85",
            "fft_rolloff_95",
            "fft_spectral_flatness",
            "fft_spectral_entropy",
            "fft_hf_slope_beta",
        ]
        for k in fft_globals:
            vals.append(self._safe_float(row.get(k)))

        # APS: 12, RPS: 64
        for i in range(12):
            vals.append(self._safe_float(row.get(f"fft_aps_{i}")))
        for i in range(64):
            vals.append(self._safe_float(row.get(f"fft_rps_{i}")))

        # Peaks: 6
        for i in range(1, 4):
            vals.append(self._safe_float(row.get(f"fft_peak{i}_r")))
            vals.append(self._safe_float(row.get(f"fft_peak{i}_val")))

        # JPEG markers: 3
        vals.append(self._safe_float(row.get("fft_jpeg_8x8_x")))
        vals.append(self._safe_float(row.get("fft_jpeg_8x8_y")))
        vals.append(self._safe_float(row.get("fft_jpeg_8x8_diag")))

        # Metadata: 5 (color_mode may be non-numeric; coerces to 0.0)
        vals.append(self._safe_float(row.get("width")))
        vals.append(self._safe_float(row.get("height")))
        vals.append(self._safe_float(row.get("color_mode")))
        vals.append(self._safe_float(row.get("resize_to")))
        vals.append(self._safe_float(row.get("do_hann")))

        arr = np.asarray(vals, dtype=np.float32)
        if arr.shape[0] != 283:
            raise ValueError(f"frequency_feature_len_expected_283 got={arr.shape[0]}")
        return arr

    def _extract_openface_features_674(self, row: Dict[str, Any]) -> np.ndarray:
        """Extract 669 OpenFace features in fixed order from named columns (excluding metadata)"""
        vals = [self._safe_float(row.get(col)) for col in self._of_columns]
        arr = np.asarray(vals, dtype=np.float32)
        if arr.shape[0] != 669:
            raise ValueError(f"openface_feature_len_expected_669 got={arr.shape[0]}")
        return arr

    @staticmethod
    def _sorted_frame_indices(frame_refs: List[Dict[str, Any]], features: List[Dict[str, Any]]) -> List[int]:
        idxs: List[int] = []
        for ref in frame_refs:
            if isinstance(ref, dict) and "frame_index" in ref:
                try:
                    idxs.append(int(ref["frame_index"]))
                except Exception:
                    pass
        for feat in features:
            if isinstance(feat, dict) and "frame_index" in feat:
                try:
                    idx = int(feat["frame_index"])
                    if idx < 0 and "frame" in feat:
                        # Handle OpenFace case: frame_index=-1, frame=1 -> use frame-1
                        try:
                            idx = int(feat["frame"]) - 1
                        except Exception:
                            pass
                    idxs.append(idx)
                except Exception:
                    pass
        return sorted(set(idxs))

    def _compute_window_start(self, frame_refs: List[Dict[str, Any]], features: List[Dict[str, Any]]) -> int:
        mode = (self._cfg.window_start_mode or "min").strip().lower()
        if mode == "zero":
            return 0
        idxs = self._sorted_frame_indices(frame_refs, features)
        return idxs[0] if idxs else 0

    def _normalize_frame_index(self, feat: Dict[str, Any]) -> int:
        """Normalize frame_index: if < 0, derive from 'frame' field (frame - 1)"""
        try:
            idx = int(feat.get("frame_index", -1))
            if idx >= 0:
                return idx
            # fallback: use frame - 1
            frame_num = int(feat.get("frame", 0))
            return max(0, frame_num - 1)
        except Exception:
            return 0

    def build_inputs(self, message: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, List[str], int]:
        frame_refs = sorted(
            message.get("frame_refs", []) or [],
            key=lambda x: int(x.get("frame_index", 0)) if isinstance(x, dict) else 0
        )
        features = message.get("features", []) or []

        # log the features 

        window_start = self._compute_window_start(frame_refs, features)
        window_indices = list(range(window_start, window_start + self._cfg.sequence_length))

        # Map frame refs
        refs_by_idx: Dict[int, Dict[str, Any]] = {}
        for ref in frame_refs:
            if not isinstance(ref, dict):
                continue
            try:
                refs_by_idx[int(ref.get("frame_index", -1))] = ref
            except Exception:
                continue

        # Map features by (normalized frame_index, source)
        feats_by_idx: Dict[int, Dict[str, Dict[str, Any]]] = {}
        for feat in features:
            if not isinstance(feat, dict):
                continue
            idx = self._normalize_frame_index(feat)
            source = str(feat.get("_source", "unknown"))
            feats_by_idx.setdefault(idx, {})[source] = feat
        

        # Build CSV sequence with padding
        last_csv_vec: Optional[np.ndarray] = None
        csv_vecs: List[np.ndarray] = []
        for idx in window_indices:
            freq_row = feats_by_idx.get(idx, {}).get("frequency")
            of_row = feats_by_idx.get(idx, {}).get("openface")

            if freq_row is None or of_row is None:
                if self._cfg.pad_sequence:
                    vec = last_csv_vec if last_csv_vec is not None else np.zeros((self._cfg.csv_features,), dtype=np.float32)
                    csv_vecs.append(vec)
                    continue
                raise ValueError(f"missing_features frame_index={idx}")

            freq_vec = self._extract_frequency_features_283(freq_row)
            of_vec = self._extract_openface_features_674(of_row)
            vec = np.concatenate([freq_vec, of_vec], axis=0).astype(np.float32, copy=False)
            if vec.shape[0] != self._cfg.csv_features:
                raise ValueError(f"csv_feature_len_expected_{self._cfg.csv_features} got={vec.shape[0]}")
            csv_vecs.append(vec)
            last_csv_vec = vec

        csv_seq = np.stack(csv_vecs, axis=0)  # (15, 957)

        if self._scaler is not None:
            flat = csv_seq.reshape(-1, self._cfg.csv_features)
            flat = self._scaler.transform(flat)
            csv_seq = flat.reshape(self._cfg.sequence_length, self._cfg.csv_features).astype(np.float32, copy=False)

        # Build image sequence with padding
        last_img: Optional[np.ndarray] = None
        imgs: List[np.ndarray] = []
        uris_used: List[str] = []
        for idx in window_indices:
            uri = None
            ref = refs_by_idx.get(idx)
            if isinstance(ref, dict):
                uri = ref.get("uri")

            img = self._frame_fetcher.fetch_frame(uri) if uri else None
            if img is None:
                if self._cfg.pad_sequence:
                    img = last_img if last_img is not None else np.zeros((self._cfg.image_size[1], self._cfg.image_size[0], 3), dtype=np.float32)
                    imgs.append(img)
                    continue
                raise ValueError(f"missing_frame frame_index={idx}")

            imgs.append(img)
            last_img = img
            if uri:
                uris_used.append(str(uri))

        img_seq = np.stack(imgs, axis=0)  # (15, 224, 224, 3)

        # Add batch dimension
        img_input = np.expand_dims(img_seq, axis=0)  # (1, 15, 224, 224, 3)
        csv_input = np.expand_dims(csv_seq, axis=0)  # (1, 15, 952)

        # Log input build summary
        logger.info(
            "input_built window_start=%d frames_used=%d uris_fetched=%d img_shape=%s csv_shape=%s "
            "img_min=%.4f img_max=%.4f img_mean=%.4f csv_min=%.4f csv_max=%.4f csv_mean=%.4f",
            window_start,
            len(window_indices),
            len(set(uris_used)),
            img_input.shape,
            csv_input.shape,
            img_input.min(),
            img_input.max(),
            img_input.mean(),
            csv_input.min(),
            csv_input.max(),
            csv_input.mean(),
        )

        return img_input, csv_input, sorted(set(uris_used)), window_start


class DeepfakeDetector:
    def __init__(self, model_path: str):
        logger.info("model_loading path=%s", model_path)
        self._model = keras.models.load_model(model_path)
        logger.info("model_loaded inputs=%s output=%s", [tuple(i.shape) for i in self._model.inputs], tuple(self._model.output.shape))

    def predict(self, img_input: np.ndarray, csv_input: np.ndarray) -> Dict[str, Any]:
        """
        Model output interpretation:
        - Training used: fake=0, real=1
        - Sigmoid output (shape=(None,1)) => probability of class 1 (REAL)
        - If prob_real >= 0.5: predict "real", else "fake"
        """
        t0 = time.time()
        pred = self._model.predict([img_input, csv_input], verbose=0)
        ms = int((time.time() - t0) * 1000)

        # Model output is sigmoid => prob_real (class 1)
        if pred.shape[-1] == 1:
            prob_real = float(pred[0, 0])
            prob_fake = 1.0 - prob_real
        else:
            # If softmax [prob_fake, prob_real]
            prob_fake = float(pred[0, 0])
            prob_real = float(pred[0, 1])

        label = "real" if prob_real >= 0.5 else "fake"
        conf = prob_real if label == "real" else prob_fake

        return {
            "prob_real": prob_real,
            "prob_fake": prob_fake,
            "pred_label": label,
            "confidence": conf,
            "inference_ms": ms,
        }


class DeepfakeConsumer:
    def __init__(self):
        self._cfg = Config()
        self._frame_fetcher = FrameFetcher(
            cache_size=self._cfg.frame_cache_size,
            timeout_s=self._cfg.frame_fetch_timeout_s,
            image_size=self._cfg.image_size,
        )
        self._input_builder = InputBuilder(config=self._cfg, frame_fetcher=self._frame_fetcher)
        self._detector = DeepfakeDetector(model_path=self._cfg.model_path)

        self._connection: Optional[pika.BlockingConnection] = None
        self._channel: Optional[pika.adapters.blocking_connection.BlockingChannel] = None
        self._stopping = False

        self.messages_processed = 0
        self.predictions_made = 0
        self.errors = 0
        
        # Socket.IO server (Option B: backend is server)
        self._sio_server: Optional[socketio.AsyncServer] = None
        self._sio_app: Optional[web.Application] = None
        self._sio_runner: Optional[web.AppRunner] = None
        self._sio_thread: Optional[threading.Thread] = None
        self._sio_loop: Optional[asyncio.AbstractEventLoop] = None
        
        if self._cfg.socketio_enabled:
            self._start_socketio_server()

    def _install_signal_handlers(self) -> None:
        def _handle(sig_num: int, _frame: Any) -> None:
            sig_name = signal.Signals(sig_num).name
            logger.info("signal_received signal=%s", sig_name)
            self.request_stop(reason=sig_name)

        signal.signal(signal.SIGINT, _handle)
        signal.signal(signal.SIGTERM, _handle)

    def _start_socketio_server(self) -> None:
        """Start Socket.IO server in background thread (Option B: backend as server)."""
        try:
            logger.info(
                "socketio_server_starting host=%s port=%s use_rooms=%s",
                self._cfg.socketio_host,
                self._cfg.socketio_port,
                self._cfg.socketio_use_rooms,
            )
            
            self._sio_thread = threading.Thread(target=self._run_socketio_server, daemon=True)
            self._sio_thread.start()
            
            # Give server time to start
            time.sleep(1)
            logger.info("socketio_server_started host=%s port=%s", self._cfg.socketio_host, self._cfg.socketio_port)
        except Exception as e:
            logger.error("socketio_server_start_failed err=%s", e, exc_info=True)
            self._sio_server = None
    
    def _run_socketio_server(self) -> None:
        """Run Socket.IO server in dedicated event loop (background thread)."""
        self._sio_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._sio_loop)
        
        try:
            self._sio_loop.run_until_complete(self._setup_socketio_server())
            self._sio_loop.run_forever()
        except Exception as e:
            logger.error("socketio_server_loop_failed err=%s", e, exc_info=True)
        finally:
            self._sio_loop.close()
    
    async def _setup_socketio_server(self) -> None:
        """Setup Socket.IO server with aiohttp."""
        # Create Socket.IO server
        self._sio_server = socketio.AsyncServer(
            async_mode="aiohttp",
            cors_allowed_origins="*",
            logger=False,
            engineio_logger=False,
        )
        
        # Setup event handlers
        @self._sio_server.event
        async def connect(sid, environ):
            query = environ.get("QUERY_STRING", "")
            session_id = None
            if "session_id=" in query:
                session_id = query.split("session_id=")[1].split("&")[0]
            
            logger.info("socketio_client_connected sid=%s session_id=%s", sid, session_id)
            
            # Store session_id in session data
            if session_id:
                await self._sio_server.save_session(sid, {"session_id": session_id})
        
        @self._sio_server.event
        async def disconnect(sid):
            session = await self._sio_server.get_session(sid)
            session_id = session.get("session_id") if session else None
            logger.info("socketio_client_disconnected sid=%s session_id=%s", sid, session_id)
        
        @self._sio_server.event
        async def join_session(sid, data):
            """Client requests to join a session room for filtering."""
            session_id = data.get("session_id")
            if session_id and self._cfg.socketio_use_rooms:
                await self._sio_server.enter_room(sid, session_id)
                logger.info("socketio_client_joined_room sid=%s session_id=%s", sid, session_id)
                await self._sio_server.save_session(sid, {"session_id": session_id})
        
        @self._sio_server.event
        async def leave_session(sid, data):
            """Client requests to leave a session room."""
            session_id = data.get("session_id")
            if session_id and self._cfg.socketio_use_rooms:
                await self._sio_server.leave_room(sid, session_id)
                logger.info("socketio_client_left_room sid=%s session_id=%s", sid, session_id)
        
        # Create aiohttp app
        self._sio_app = web.Application()
        self._sio_server.attach(self._sio_app)
        
        # Start server
        self._sio_runner = web.AppRunner(self._sio_app)
        await self._sio_runner.setup()
        site = web.TCPSite(self._sio_runner, self._cfg.socketio_host, self._cfg.socketio_port)
        await site.start()
        
        logger.info(
            "socketio_server_ready host=%s port=%s",
            self._cfg.socketio_host,
            self._cfg.socketio_port,
        )

    def emit_prediction_to_mobile(
        self, session_id: str, batch_id: str, window_start: int, pred: Dict[str, Any]
    ) -> bool:
        """Emit prediction result to mobile clients via Socket.IO server (Option B)."""
        if not self._cfg.socketio_enabled or self._sio_server is None or self._sio_loop is None:
            return True

        payload = {
            "session_id": session_id,
            "batch_id": batch_id,
            "window_start": window_start,
            "label": pred["pred_label"],
            "prob_real": pred["prob_real"],
            "prob_fake": pred["prob_fake"],
            "confidence": pred["confidence"],
            "inference_ms": pred["inference_ms"],
            "timestamp": time.time(),
        }

        # Log the payload that will be emitted
        try:
            logger.debug("mobile_emit_payload session_id=%s batch_id=%s payload=%s", session_id, batch_id, json.dumps(payload))
        except Exception:
            logger.debug("mobile_emit_payload session_id=%s batch_id=%s payload_unserializable", session_id, batch_id)

        # Schedule emit in Socket.IO event loop
        future = asyncio.run_coroutine_threadsafe(
            self._emit_to_clients(session_id, payload),
            self._sio_loop
        )
        
        try:
            future.result(timeout=1.0)  # Wait up to 1s for emit
            logger.info(
                "socketio_emit_success event=%s session_id=%s batch_id=%s label=%s conf=%.4f",
                self._cfg.socketio_event,
                session_id,
                batch_id,
                pred["pred_label"],
                pred["confidence"],
            )
            return True
        except Exception as e:
            logger.warning(
                "socketio_emit_error event=%s session_id=%s batch_id=%s err=%s",
                self._cfg.socketio_event,
                session_id,
                batch_id,
                e,
            )
            return False
    
    async def _emit_to_clients(self, session_id: str, payload: Dict[str, Any]) -> None:
        """Async emit to Socket.IO clients (runs in server event loop)."""
        if self._cfg.socketio_use_rooms:
            # Emit only to clients in the session room
            await self._sio_server.emit(
                self._cfg.socketio_event,
                payload,
                room=session_id,
            )
            logger.debug("socketio_emit_to_room room=%s", session_id)
        else:
            # Broadcast to all connected clients
            await self._sio_server.emit(
                self._cfg.socketio_event,
                payload,
            )
            logger.debug("socketio_emit_broadcast")

    def request_stop(self, reason: str) -> None:
        if self._stopping:
            return
        self._stopping = True

        if self._connection is not None and self._channel is not None:
            try:
                self._connection.add_callback_threadsafe(self._channel.stop_consuming)
                logger.info("stop_requested reason=%s", reason)
                return
            except Exception as e:
                logger.warning("stop_request_failed reason=%s err=%s", reason, e)

        logger.info("stop_requested_no_channel reason=%s", reason)

    def connect_rabbitmq(self) -> None:
        logger.info("rabbitmq_connecting url=%s", self._cfg.rabbitmq_url)
        params = pika.URLParameters(self._cfg.rabbitmq_url)
        self._connection = pika.BlockingConnection(params)
        self._channel = self._connection.channel()

        self._channel.queue_declare(queue=self._cfg.rabbitmq_queue, durable=True)
        self._channel.basic_qos(prefetch_count=1)

        logger.info("rabbitmq_connected queue=%s", self._cfg.rabbitmq_queue)

    def process_message(self, ch, method, properties, body: bytes) -> None:
        delivery_tag = getattr(method, "delivery_tag", None)

        try:
            message = json.loads(body)
        except Exception as e:
            logger.error("json_parse_failed delivery_tag=%s err=%s", delivery_tag, e)
            try:
                ch.basic_ack(delivery_tag=delivery_tag)
                logger.info("ACK delivery_tag=%s reason=json_parse_failed", delivery_tag)
            except Exception as ack_e:
                logger.error("ack_failed delivery_tag=%s err=%s", delivery_tag, ack_e)
            self.errors += 1
            return

        session_id = message.get("session_id", "unknown")
        batch_id = message.get("batch_id", "unknown")
        frame_count = message.get("frame_count", None)
        features = message.get("features", []) or []
        frame_refs = message.get("frame_refs", []) or []

        logger.info(
            "RECV delivery_tag=%s session_id=%s batch_id=%s frame_count=%s features_count=%s frame_refs_count=%s",
            delivery_tag,
            session_id,
            batch_id,
            frame_count,
            len(features),
            len(frame_refs),
        )

        # Validate message has features
        if len(features) == 0:
            logger.error(
                "invalid_message_no_features delivery_tag=%s session_id=%s batch_id=%s frame_count=%s -> NACK(requeue=%s)",
                delivery_tag,
                session_id,
                batch_id,
                frame_count,
                self._cfg.requeue_on_error,
            )
            try:
                ch.basic_nack(delivery_tag=delivery_tag, requeue=self._cfg.requeue_on_error)
                logger.info("NACK delivery_tag=%s reason=no_features requeue=%s", delivery_tag, self._cfg.requeue_on_error)
            except Exception as nack_e:
                logger.error("nack_failed delivery_tag=%s err=%s", delivery_tag, nack_e)
            self.errors += 1
            return

        try:
            img_input, csv_input, uris_used, window_start = self._input_builder.build_inputs(message)
            pred = self._detector.predict(img_input, csv_input)

            logger.info(
                "prediction delivery_tag=%s session_id=%s batch_id=%s window_start=%s label=%s prob_real=%.6f prob_fake=%.6f conf=%.6f ms=%s",
                delivery_tag,
                session_id,
                batch_id,
                window_start,
                pred["pred_label"],
                pred["prob_real"],
                pred["prob_fake"],
                pred["confidence"],
                pred["inference_ms"],
            )

            # Emit prediction to mobile client
            self.emit_prediction_to_mobile(session_id, batch_id, window_start, pred)

            ch.basic_ack(delivery_tag=delivery_tag)
            logger.info("ACK delivery_tag=%s session_id=%s batch_id=%s", delivery_tag, session_id, batch_id)

            self.messages_processed += 1
            self.predictions_made += 1

            if self._cfg.delete_frames and uris_used:
                self._frame_fetcher.delete_batch(uris_used, timeout_s=self._cfg.delete_timeout_s)
                logger.info(
                    "frames_delete_attempted delivery_tag=%s session_id=%s batch_id=%s count=%d",
                    delivery_tag,
                    session_id,
                    batch_id,
                    len(uris_used),
                )

        except Exception as e:
            self.errors += 1
            requeue = self._cfg.requeue_on_error
            logger.error(
                "processing_failed delivery_tag=%s session_id=%s batch_id=%s requeue=%s err=%s",
                delivery_tag,
                session_id,
                batch_id,
                requeue,
                e,
                exc_info=True,
            )
            try:
                ch.basic_nack(delivery_tag=delivery_tag, requeue=requeue)
                logger.info(
                    "NACK delivery_tag=%s session_id=%s batch_id=%s requeue=%s",
                    delivery_tag,
                    session_id,
                    batch_id,
                    requeue,
                )
            except Exception as nack_e:
                logger.error("nack_failed delivery_tag=%s err=%s", delivery_tag, nack_e)

    def start(self) -> None:
        self._install_signal_handlers()
        self.connect_rabbitmq()

        assert self._channel is not None
        logger.info("consumer_started queue=%s (Ctrl+C to stop)", self._cfg.rabbitmq_queue)

        self._channel.basic_consume(queue=self._cfg.rabbitmq_queue, on_message_callback=self.process_message)

        try:
            self._channel.start_consuming()
        finally:
            try:
                if self._channel is not None and self._channel.is_open:
                    try:
                        self._channel.stop_consuming()
                    except Exception:
                        pass
            finally:
                if self._connection is not None and self._connection.is_open:
                    try:
                        self._connection.close()
                    except Exception:
                        pass

            # Stop Socket.IO server
            if self._sio_server is not None and self._sio_loop is not None:
                try:
                    # Stop server event loop
                    self._sio_loop.call_soon_threadsafe(self._sio_loop.stop)
                    if self._sio_thread is not None:
                        self._sio_thread.join(timeout=5)
                    logger.info("socketio_server_stopped")
                except Exception as e:
                    logger.warning("socketio_server_stop_failed err=%s", e)

            logger.info(
                "consumer_stopped processed=%d predictions=%d errors=%d",
                self.messages_processed,
                self.predictions_made,
                self.errors,
            )


if __name__ == "__main__":
    DeepfakeConsumer().start()
