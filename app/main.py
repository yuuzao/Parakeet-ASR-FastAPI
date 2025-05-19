import asyncio
import base64
import io
import logging
import os
import shutil
import tempfile

import nemo.collections.asr as nemo_asr
import torch
import torchaudio

# Load environment variables from .env file if it exists
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.websockets import WebSocketState

from models import TranscriptionResponse, Word, Segment, Segments

# import time # No longer needed as timing logic has been removed/commented out


load_dotenv()

# Configure logging
log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("parakeet-asr")
logger.info(f"Logging configured with level: {log_level_str}")

# model configuration
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 1))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 0))
CHUNK_LENGTH = float(os.getenv("TRANSCRIBE_CHUNK_LEN", 30))
OVERLAP = float(os.getenv("TRANSCRIBE_OVERLAP", 5))
MODEL_SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", 16000))
PORT = int(os.getenv("PORT", 8777))  # Add PORT from .env
logger.info(
    f"Model config: BATCH_SIZE: {BATCH_SIZE}, NUM_WORKERS: {NUM_WORKERS}, CHUNK_LENGTH: {CHUNK_LENGTH}, OVERLAP: {OVERLAP}, MODEL_SAMPLE_RATE: {MODEL_SAMPLE_RATE}"
)

# FAST API app with CORS
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    # If running directly (not in Docker), the static dir might be in a different location
    static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
    if not os.path.exists(static_dir):
        os.makedirs(static_dir, exist_ok=True)
        logger.warning(f"Created static directory at {static_dir}")

app.mount("/static", StaticFiles(directory=static_dir), name="static")


# Root endpoint to serve index.html
@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    else:
        logger.error(f"Index file not found at {index_path}")
        return HTMLResponse(
            content="<html><body><h1>Parakeet ASR Server</h1><p>UI not available. index.html not found.</p></body></html>"
        )


# load model
asr_model = None
try:
    logger.info("Loading ASR model...")
    asr_model = nemo_asr.models.ASRModel.from_pretrained(
        model_name="nvidia/parakeet-tdt-0.6b-v2"
    )
    asr_model.preprocessor.featurizer.dither = 0.0
    # ====== 新增：启用词级别时间戳 ======
    try:
        from omegaconf import open_dict

        decoding_cfg = asr_model.cfg.decoding
        with open_dict(decoding_cfg):
            decoding_cfg.preserve_alignments = True
            decoding_cfg.compute_timestamps = True
        asr_model.change_decoding_strategy(decoding_cfg)
        logger.info("ASR model decoding config updated: word timestamps enabled.")
    except Exception as e:
        logger.warning(f"Failed to set word timestamp config: {e}")
    # ====== 新增结束 ======
    logger.info("ASR model loaded successfully.")
except Exception as e:
    logger.critical(f"FATAL: Could not load ASR model. Error: {e}")
    asr_model = None


# This function processes a 1D audio tensor chunk.
# It is called by the endpoints after the full audio file has been decoded
# and chunked according to server-side chunk_len/overlap logic.
def run_asr_on_tensor_chunk(
    audio_chunk_tensor: torch.Tensor,  # Expected to be 1D, mono, at MODEL_SAMPLE_RATE
    chunk_time_offset: float,  # Absolute start time of this chunk in the original audio
) -> tuple[list[Segment], list[str]]:
    """
    Transcribes a 1D audio tensor chunk and adjusts timestamps.

    Args:
        audio_chunk_tensor: A 1D PyTorch tensor of the audio chunk.
                            The caller must ensure it's correctly formatted (mono, MODEL_SAMPLE_RATE).
        chunk_time_offset: The absolute start time (in seconds) of this audio_chunk_tensor
                           relative to the beginning of the original full audio stream.

    Returns:
        A tuple containing:
        - processed_segments (list[dict]): A list of segment dictionaries with absolute timestamps.
        - chunk_full_text_parts (list[str]): A list containing the transcribed text of the chunk (usually one item).
    """
    if not asr_model:
        # This check is still critical as the model might fail to load globally.
        logger.error("run_asr_on_tensor_chunk: ASR model is not loaded.")
        return [], []

    # --- Pre-condition Check (Development/Debug Aid) ---
    # The caller is responsible for providing a 1D tensor. This assertion helps catch caller errors during development.
    # In a highly optimized production environment, this could be removed if the contract is strictly followed.
    assert (
        audio_chunk_tensor.ndim == 1
    ), f"ERROR: run_asr_on_tensor_chunk received tensor with ndim={audio_chunk_tensor.ndim}. Expected 1D."
    assert (
        audio_chunk_tensor.numel() > 0
    ), "ERROR: run_asr_on_tensor_chunk received an empty audio tensor."

    try:
        # Perform ASR transcription on the provided 1D tensor chunk.
        # `num_workers=0` is appropriate as this function is typically run in `asyncio.to_thread`.
        output_hypotheses: list = asr_model.transcribe(
            audio=[audio_chunk_tensor],  # NeMo expects a list of 1D tensors.
            batch_size=BATCH_SIZE,  # We process one chunk at a time here.
            return_hypotheses=True,  # Request Hypothesis objects for rich output.
            timestamps=True,  # Request timestamp information.
            verbose=False,  # Suppress NeMo's internal transcription logging for this call.
            num_workers=NUM_WORKERS,
        )

        processed_segments: list[Segment] = []
        chunk_full_text_parts: list[str] = []  # This list will store the full text transcription parts for the chunk.

        # `asr_model.transcribe` with batch_size=1 and return_hypotheses=True
        # is expected to return a list containing exactly one Hypothesis object.
        if not (
            output_hypotheses
            and isinstance(output_hypotheses, list)
            and len(output_hypotheses) == 1
        ):
            logger.warning(
                f"run_asr_on_tensor_chunk: Unexpected output format from ASR model: {type(output_hypotheses)}"
            )
            return [], []

        hypothesis = output_hypotheses[0]

        # Extract the full text for this chunk if available.
        if hasattr(hypothesis, "text") and hypothesis.text:
            transcribed_text = hypothesis.text.strip()
            if transcribed_text:
                chunk_full_text_parts.append(transcribed_text)

        # Attempt to get word-level timestamps for the entire chunk
        word_timestamps_for_chunk = []
        if (
            hasattr(hypothesis, "timestamp")
            and hypothesis.timestamp
            and "word" in hypothesis.timestamp
        ):
            word_timestamps_for_chunk = []
            for word_meta in hypothesis.timestamp["word"]:
                word = Word(
                    word=word_meta.get("word", ""),
                    start=word_meta.get("start") + chunk_time_offset,
                    end=word_meta.get("end") + chunk_time_offset,
                    start_offset=word_meta.get("start_offset"),
                    end_offset=word_meta.get("end_offset"),
                )
                word_timestamps_for_chunk.append(word)
            word_timestamps_for_chunk.sort(key=lambda x: x.start_offset)
            logger.info(
                f"使用 hypothesis.timestamp['word']，长度: {len(word_timestamps_for_chunk)}"
            )
            logger.debug(f"timestamp['word'] 内容: {word_timestamps_for_chunk}")
        else:
            logger.warning("没有找到词级别时间戳（timestamp['word']）！")

        # Extract segment-level timestamps if available.
        if hasattr(hypothesis, "timestamp") and hypothesis.timestamp:
            segment_metadata_list: list[Segment] = []
            for seg_meta in hypothesis.timestamp.get("segment", []):
                seg = Segment(
                    start=seg_meta.get("start") + chunk_time_offset,
                    end=seg_meta.get("end") + chunk_time_offset,
                    start_offset=seg_meta.get("start_offset"),
                    end_offset=seg_meta.get("end_offset"),
                    text=seg_meta.get("segment", "").strip(),
                    words=[]
                )
                segment_metadata_list.append(seg)
                segment_metadata_list.sort(key=lambda x: x.start_offset)

            for seg_meta in segment_metadata_list:
                for word in word_timestamps_for_chunk:
                    if word.start >= seg_meta.start and word.end <= seg_meta.end:
                        seg_meta.words.append(word)

                processed_segments.append(seg_meta)

        return processed_segments, chunk_full_text_parts  # Return chunk_full_text_parts

    except Exception as e:
        # Log the error and return empty lists to allow the calling process to continue if appropriate.
        logger.error(
            f"run_asr_on_tensor_chunk: Exception during ASR processing: {str(e)}"
        )
        import traceback

        traceback.print_exc()
        return [], []


@app.post("/v1/audio/transcriptions", response_model=TranscriptionResponse)
async def transcribe_rest(file: UploadFile = File(...)):
    """
    Handles audio transcription via REST API.
    The uploaded audio file is saved temporarily, then processed in chunks
    with overlap using NeMo ASR. Each chunk is saved as a temporary WAV file
    and passed to the ASR model.
    """
    if not asr_model:  # Use the global asr_model
        logger.error("transcribe_rest: ASR model not available.")
        return JSONResponse(
            status_code=503, content={"error": "ASR model not available."}
        )

    # Path for the initially uploaded (full) temporary file
    uploaded_full_temp_file_path = ""
    try:
        # 1. Save the uploaded file to a temporary location
        # Using delete=False because we need the path for torchaudio.load
        # and will manually delete it in the finally block.
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".tmp"
        ) as tmp_full_audio:  # Suffix can be generic
            shutil.copyfileobj(file.file, tmp_full_audio)
            uploaded_full_temp_file_path = tmp_full_audio.name

        # 2. Load and preprocess the full waveform
        try:
            waveform_full, sr_orig = torchaudio.load(uploaded_full_temp_file_path)
        except Exception as load_err:
            logger.error(
                f"transcribe_rest: Failed to load audio from {uploaded_full_temp_file_path}: {load_err}"
            )
            raise  # Re-raise to be caught by the outer try-except

        if sr_orig != MODEL_SAMPLE_RATE:
            waveform_full = torchaudio.functional.resample(
                waveform_full, orig_freq=sr_orig, new_freq=MODEL_SAMPLE_RATE
            )

        # Ensure waveform is mono and has a batch-like dimension [1, Time] for consistent slicing.
        # Most ASR models operate on mono.
        if waveform_full.ndim > 1 and waveform_full.shape[0] > 1:  # Multichannel
            waveform_full = waveform_full.mean(
                dim=0, keepdim=True
            )  # Convert to mono [1, Time]
        elif waveform_full.ndim == 1:  # Mono but 1D
            waveform_full = waveform_full.unsqueeze(
                0
            )  # Add channel/batch dim: [1, Time]
        # Now waveform_full is expected to be [1, Time]

        if waveform_full.shape[1] == 0:  # Check for empty audio after processing
            logger.info(
                "transcribe_rest: Audio content is empty after loading and preprocessing."
            )
            return JSONResponse(
                content={
                    "text": "",
                    "segments": [],
                    "language": "en",
                    "transcription_time": 0.0,
                }
            )

        total_duration_seconds = waveform_full.shape[1] / MODEL_SAMPLE_RATE

        # 3. Process audio in chunks using a sliding window
        current_processing_time_seconds = 0.0  # Tracks the start of the main (non-overlapped) part of the current window
        all_segments: list[Segment] = []
        while current_processing_time_seconds < total_duration_seconds:
            # Define the actual audio slice considering overlap.
            # `actual_chunk_start_seconds` can go before `current_processing_time_seconds` due to overlap.
            actual_chunk_start_seconds = max(
                0, current_processing_time_seconds - OVERLAP
            )
            # `actual_chunk_end_seconds` is the end of the full window (main part + right overlap part).
            actual_chunk_end_seconds = min(
                total_duration_seconds, current_processing_time_seconds + CHUNK_LENGTH
            )

            # Convert times to sample indices
            start_sample = int(actual_chunk_start_seconds * MODEL_SAMPLE_RATE)
            end_sample = int(actual_chunk_end_seconds * MODEL_SAMPLE_RATE)

            # If the calculated chunk is empty or invalid, stop.
            if start_sample >= end_sample:
                break

            # Slice the waveform_full (which is [1, Time]) to get a [1, chunk_Time] tensor
            chunk_slice_2d = waveform_full[:, start_sample:end_sample]

            # Squeeze to make it 1D [chunk_Time] for run_asr_on_tensor_chunk
            audio_chunk_for_asr = chunk_slice_2d.squeeze(0)

            if audio_chunk_for_asr.numel() == 0:
                current_processing_time_seconds += CHUNK_LENGTH - OVERLAP
                continue

            segments_from_chunk, _ = await asyncio.to_thread(
                run_asr_on_tensor_chunk, audio_chunk_for_asr, actual_chunk_start_seconds
            )
            all_segments.extend(segments_from_chunk)

            current_processing_time_seconds += CHUNK_LENGTH - OVERLAP

        # Consolidate all word segments from all_segments into a single list
        all_word_segments_for_response = []
        for seg in all_segments:
            all_word_segments_for_response.extend(seg.words)

        final_text = " ".join(s.text for s in all_segments).strip()

        segments = Segments(segments=all_segments, word_segments=all_word_segments_for_response)

        result = {
            "task": "transcribe",  # Added task field
            "language": "en",  # Assuming English, can be made dynamic if needed
            "duration": round(
                total_duration_seconds, 3
            ),  # Added duration of the original audio
            "text": final_text,
            "segments": segments,
        }
        transcription_response = TranscriptionResponse(**result)
        return transcription_response

    except Exception as e:
        logger.error(f"transcribe_rest: Unhandled exception: {str(e)}")
        import traceback

        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to process audio.", "detail": str(e)},
        )
    finally:
        if uploaded_full_temp_file_path and os.path.exists(
            uploaded_full_temp_file_path
        ):
            os.remove(
                uploaded_full_temp_file_path
            )  # Delete the initially uploaded full temp file


if __name__ == "__main__":
    import uvicorn

    if not asr_model:
        logger.critical("Cannot start server: ASR Model failed to load.")
    else:
        logger.info(f"Starting server on port {PORT}. Model Rate: {MODEL_SAMPLE_RATE}")
        uvicorn.run(app, host="0.0.0.0", port=PORT)
