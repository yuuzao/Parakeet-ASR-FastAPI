import os
import io
import time
import shutil
import tempfile
import base64
import asyncio
import logging

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import torch
import torchaudio
import nemo.collections.asr as nemo_asr

# Load environment variables from .env file if it exists
from dotenv import load_dotenv
load_dotenv()

# Configure logging
log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
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
logger.info(f"Model config: BATCH_SIZE: {BATCH_SIZE}, NUM_WORKERS: {NUM_WORKERS}, CHUNK_LENGTH: {CHUNK_LENGTH}, OVERLAP: {OVERLAP}, MODEL_SAMPLE_RATE: {MODEL_SAMPLE_RATE}")

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
        return HTMLResponse(content="<html><body><h1>Parakeet ASR Server</h1><p>UI not available. index.html not found.</p></body></html>")

# load model
asr_model = None 
try:
    logger.info("Loading ASR model...")
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")
    asr_model.preprocessor.featurizer.dither = 0.0  
    logger.info("ASR model loaded successfully.")
except Exception as e:
    logger.critical(f"FATAL: Could not load ASR model. Error: {e}")
    asr_model = None 


# This function processes a 1D audio tensor chunk.
# It is called by the endpoints after the full audio file has been decoded
# and chunked according to server-side chunk_len/overlap logic.
def run_asr_on_tensor_chunk(
    audio_chunk_tensor: torch.Tensor,  # Expected to be 1D, mono, at MODEL_SAMPLE_RATE
    chunk_time_offset: float           # Absolute start time of this chunk in the original audio
) -> tuple[list[dict], list[str]]:
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
    assert audio_chunk_tensor.ndim == 1, \
        f"ERROR: run_asr_on_tensor_chunk received tensor with ndim={audio_chunk_tensor.ndim}. Expected 1D."
    assert audio_chunk_tensor.numel() > 0, \
        "ERROR: run_asr_on_tensor_chunk received an empty audio tensor."

    try:
        # Perform ASR transcription on the provided 1D tensor chunk.
        # `num_workers=0` is appropriate as this function is typically run in `asyncio.to_thread`.
        output_hypotheses: list = asr_model.transcribe(
            audio=[audio_chunk_tensor],  # NeMo expects a list of 1D tensors.
            batch_size=BATCH_SIZE,       # We process one chunk at a time here.
            return_hypotheses=True,      # Request Hypothesis objects for rich output.
            timestamps=True,             # Request timestamp information.
            verbose=False,               # Suppress NeMo's internal transcription logging for this call.
            num_workers=NUM_WORKERS
        )
        
        processed_segments = []
        chunk_text_list = [] # Renamed from chunk_full_text_parts for clarity

        # `asr_model.transcribe` with batch_size=1 and return_hypotheses=True
        # is expected to return a list containing exactly one Hypothesis object.
        if not (output_hypotheses and isinstance(output_hypotheses, list) and len(output_hypotheses) == 1):
            logger.warning(f"run_asr_on_tensor_chunk: Unexpected output format from ASR model: {type(output_hypotheses)}")
            # If the model for some reason didn't transcribe or returned an unexpected format,
            # we return empty results to prevent further errors.
            return [], []

        hypothesis = output_hypotheses[0]

        # Extract the full text for this chunk if available.
        # The text might be None or an empty string for silent/unintelligible audio.
        if hasattr(hypothesis, 'text') and hypothesis.text: # Check for existence and non-emptiness
            # .strip() to remove leading/trailing whitespace from model's raw output.
            transcribed_text = hypothesis.text.strip()
            if transcribed_text: # Only add if there's actual content after stripping
                chunk_text_list.append(transcribed_text)

        # Extract segment-level timestamps if available.
        # The `hypothesis.timestamp` attribute (a dict) is expected if `timestamps=True`.
        # The "segment" key within this dict contains a list of segment metadata.
        if hasattr(hypothesis, 'timestamp') and hypothesis.timestamp:
            segment_metadata_list = hypothesis.timestamp.get("segment", []) # Default to empty list
            
            for seg_meta in segment_metadata_list:
                seg_text = seg_meta.get("segment", "").strip()
                # Skip segments that are empty after stripping whitespace.
                if not seg_text:
                    continue

                # Timestamps from the model (`seg_meta["start"]`, `seg_meta["end"]`) are relative
                # to the beginning of the `audio_chunk_tensor` that was transcribed.
                # Add `chunk_time_offset` to make them absolute to the original full audio.
                abs_start_time = round(seg_meta["start"] + chunk_time_offset, 3)
                abs_end_time = round(seg_meta["end"] + chunk_time_offset, 3)

                # Construct the segment dictionary.
                # Other fields are placeholders matching OpenAI's format, can be omitted if not used.
                processed_segments.append({
                    "start": abs_start_time,
                    "end": abs_end_time,
                    "text": seg_text,
                    # Placeholder fields:
                    "id": 0, # ID will be assigned by the caller
                    "seek": 0, 
                    "tokens": [], 
                    "temperature": 0.0,
                    "avg_logprob": None, 
                    "compression_ratio": None, 
                    "no_speech_prob": None
                })
        
        return processed_segments, chunk_text_list
    
    except Exception as e:
        # Log the error and return empty lists to allow the calling process to continue if appropriate.
        logger.error(f"run_asr_on_tensor_chunk: Exception during ASR processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return [], []


@app.post("/v1/audio/transcriptions")
async def transcribe_rest(file: UploadFile = File(...)):
    """
    Handles audio transcription via REST API.
    The uploaded audio file is saved temporarily, then processed in chunks
    with overlap using NeMo ASR. Each chunk is saved as a temporary WAV file
    and passed to the ASR model.
    """
    if not asr_model: # Use the global asr_model
        logger.error("transcribe_rest: ASR model not available.")
        return JSONResponse(status_code=503, content={"error": "ASR model not available."})

    # Path for the initially uploaded (full) temporary file
    uploaded_full_temp_file_path = "" 
    try:
        # 1. Save the uploaded file to a temporary location
        # Using delete=False because we need the path for torchaudio.load
        # and will manually delete it in the finally block.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp_full_audio: # Suffix can be generic
            shutil.copyfileobj(file.file, tmp_full_audio)
            uploaded_full_temp_file_path = tmp_full_audio.name
        
        # 2. Load and preprocess the full waveform
        try:
            waveform_full, sr_orig = torchaudio.load(uploaded_full_temp_file_path)
        except Exception as load_err:
            logger.error(f"transcribe_rest: Failed to load audio from {uploaded_full_temp_file_path}: {load_err}")
            raise # Re-raise to be caught by the outer try-except

        if sr_orig != MODEL_SAMPLE_RATE:
            waveform_full = torchaudio.functional.resample(waveform_full, orig_freq=sr_orig, new_freq=MODEL_SAMPLE_RATE)
        
        # Ensure waveform is mono and has a batch-like dimension [1, Time] for consistent slicing.
        # Most ASR models operate on mono.
        if waveform_full.ndim > 1 and waveform_full.shape[0] > 1: # Multichannel
            waveform_full = waveform_full.mean(dim=0, keepdim=True) # Convert to mono [1, Time]
        elif waveform_full.ndim == 1: # Mono but 1D
             waveform_full = waveform_full.unsqueeze(0) # Add channel/batch dim: [1, Time]
        # Now waveform_full is expected to be [1, Time]

        if waveform_full.shape[1] == 0: # Check for empty audio after processing
            logger.info("transcribe_rest: Audio content is empty after loading and preprocessing.")
            return JSONResponse(content={"text": "", "segments": [], "language": "en", "transcription_time": 0.0})

        total_duration_seconds = waveform_full.shape[1] / MODEL_SAMPLE_RATE
        
        # 3. Process audio in chunks using a sliding window
        start_time_processing = time.time()
        current_processing_time_seconds = 0.0 # Tracks the start of the main (non-overlapped) part of the current window
        all_segments = []
        
        # The loop continues as long as there's a new main window to process.
        while current_processing_time_seconds < total_duration_seconds:
            # Define the actual audio slice considering overlap.
            # `actual_chunk_start_seconds` can go before `current_processing_time_seconds` due to overlap.
            actual_chunk_start_seconds = max(0, current_processing_time_seconds - OVERLAP)
            # `actual_chunk_end_seconds` is the end of the full window (main part + right overlap part).
            actual_chunk_end_seconds = min(total_duration_seconds, current_processing_time_seconds + CHUNK_LENGTH)

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
                current_processing_time_seconds += (CHUNK_LENGTH - OVERLAP)
                continue

            segments_from_chunk, _ = await asyncio.to_thread( 
                run_asr_on_tensor_chunk,
                audio_chunk_for_asr,
                actual_chunk_start_seconds
            )

            # Process the transcription output for this chunk
            for seg_meta in segments_from_chunk: # segments_from_chunk already contains dicts with absolute times
                # Filtering logic (same as before)
                if seg_meta["start"] >= current_processing_time_seconds or not all_segments:
                    all_segments.append({
                        "id": len(all_segments), 
                        "start": seg_meta["start"], 
                        "end": seg_meta["end"], 
                        "text": seg_meta["text"],
                        # Placeholder fields:
                        "seek": 0, 
                        "tokens": [], 
                        "temperature": 0.0,
                        "avg_logprob": None, 
                        "compression_ratio": None, 
                        "no_speech_prob": None
                    })
            
            current_processing_time_seconds += (CHUNK_LENGTH - OVERLAP)
        
        transcription_duration_seconds = round(time.time() - start_time_processing, 3)
        final_text = " ".join(s['text'] for s in all_segments).strip()
        result = {
            "text": final_text, "segments": all_segments, "language": "en", 
            "transcription_time": transcription_duration_seconds
        }
        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"transcribe_rest: Unhandled exception: {str(e)}")
        import traceback; traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": "Failed to process audio.", "detail": str(e)})
    finally:
        if uploaded_full_temp_file_path and os.path.exists(uploaded_full_temp_file_path):
            os.remove(uploaded_full_temp_file_path) # Delete the initially uploaded full temp file


@app.websocket("/v1/audio/transcriptions/ws")
async def websocket_transcribe(websocket: WebSocket):
    """
    Handles audio transcription via WebSocket.
    The client streams the entire audio file. The server accumulates all data,
    decodes the full audio, then processes it in chunks with overlap
    (similar to the REST endpoint's logic) by calling `run_asr_on_tensor_chunk`.
    """
    await websocket.accept()
    if not asr_model: # Global asr_model
        logger.error("websocket_transcribe: ASR model not available.")
        await websocket.send_json({"error": "ASR model not available."})
        await websocket.close(code=1011)
        return

    main_audio_buffer = bytearray()  # Buffer for raw file bytes from client
    client_config = {}               # For storing client-sent metadata (e.g., original format)

    is_connected = True # Flag to control the receiving loop
    try:
        # 1. Receive client configuration (optional, primarily for logging)
        try:
            config_message = await asyncio.wait_for(websocket.receive_json(), timeout=10.0)
            client_config.update(config_message)
            logger.info(f"WebSocket (/ws) client reported config: {client_config}")
        except asyncio.TimeoutError:
            logger.info("WebSocket (/ws): Client configuration timeout. Proceeding.")
        except Exception as e:
            logger.info(f"WebSocket (/ws): Error receiving client configuration: {e}. Proceeding.")
        
        # 2. Accumulate all audio file bytes from the client stream
        logger.info("WebSocket (/ws): Waiting for audio data stream from client...")
        while is_connected:
            try:
                if websocket.application_state != WebSocketState.CONNECTED:
                    is_connected = False
                    break
                
                message = await asyncio.wait_for(websocket.receive(), timeout=60.0) # Timeout for individual messages
                
                if "bytes" in message:
                    main_audio_buffer.extend(message["bytes"])
                elif "text" in message:
                    if client_config.get("format") == "base64": # Handle base64 if client specifies
                        try:
                            main_audio_buffer.extend(base64.b64decode(message["text"]))
                        except Exception as b64e:
                            logger.warning(f"WebSocket (/ws): Base64 decode error: {b64e}")
                            # Decide if this is a fatal error or if we should continue
                    elif message["text"].upper() == "END":
                        logger.info(f"WebSocket (/ws): END signal received. Total bytes: {len(main_audio_buffer)}")
                        is_connected = False # Signal to stop receiving
                        break 
            except asyncio.TimeoutError:
                logger.warning("WebSocket (/ws): Timeout waiting for message. Assuming stream ended or client stalled.")
                is_connected = False # Stop receiving
                break
            except WebSocketDisconnect:
                logger.info("WebSocket (/ws): Client disconnected during data accumulation.")
                is_connected = False # Stop receiving
                break
            except Exception as e:
                logger.error(f"WebSocket (/ws): Exception in receive loop: {str(e)}")
                import traceback; traceback.print_exc()
                is_connected = False # Stop receiving
                break
        
        if not main_audio_buffer:
            logger.info("WebSocket (/ws): No audio data received.")
            if websocket.application_state == WebSocketState.CONNECTED: # Check before sending
                await websocket.send_json({"error": "No audio data received", "type": "error"})
            return

        # --- Start of full audio processing logic (after all bytes received) ---
        logger.info("WebSocket (/ws): Starting full audio processing.")
        processing_start_time = time.time()
        all_segments_sent_to_client = [] # Tracks segments sent for final summary
        
        try:
            # 3. Decode the entire accumulated audio stream
            audio_io_buffer = io.BytesIO(main_audio_buffer)
            full_waveform, sr_original = torchaudio.load(audio_io_buffer)
            main_audio_buffer.clear() # Release memory as soon as possible
            logger.info(f"WebSocket (/ws): Audio decoded. Original SR={sr_original}, Shape={full_waveform.shape}")

            # 4. Preprocess: Resample to model's sample rate and convert to mono
            if sr_original != MODEL_SAMPLE_RATE:
                full_waveform = torchaudio.functional.resample(full_waveform, orig_freq=sr_original, new_freq=MODEL_SAMPLE_RATE)
            
            if full_waveform.ndim > 1 and full_waveform.shape[0] > 1: # Multichannel
                full_waveform = full_waveform.mean(dim=0) # Convert to mono [Time]
            elif full_waveform.ndim == 2 and full_waveform.shape[0] == 1: # Already mono [1, Time]
                full_waveform = full_waveform.squeeze(0) # Make it 1D [Time]
            # Now, full_waveform is expected to be a 1D tensor [Time]

            if full_waveform.numel() == 0:
                logger.info("WebSocket (/ws): Audio content is empty after decoding/preprocessing.")
                if websocket.application_state == WebSocketState.CONNECTED:
                    await websocket.send_json({
                        "text": "", "segments": [], "language": "en",
                        "transcription_time": 0.0, "total_segments": 0,
                        "final_duration_processed_seconds": 0.0, "type": "final_transcription"
                    })
                return

            total_audio_duration_seconds = full_waveform.shape[0] / MODEL_SAMPLE_RATE
            current_processing_window_start_seconds = 0.0 # Start of the main (non-overlapped) part of the window
            
            logger.info(f"WebSocket (/ws): Server-side chunking. Total Duration: {total_audio_duration_seconds:.2f}s. ChunkLen: {CHUNK_LENGTH}s, Overlap: {OVERLAP}s")

            # 5. Process audio in chunks using a sliding window
            while current_processing_window_start_seconds < total_audio_duration_seconds:
                if websocket.application_state != WebSocketState.CONNECTED:
                    logger.info("WebSocket (/ws): Client disconnected during chunk processing.")
                    break
                
                # Calculate the actual slice for ASR, including overlap
                # CHUNK_LENGTH and OVERLAP are from your global environment variables
                actual_asr_chunk_start_seconds = max(0, current_processing_window_start_seconds - OVERLAP)
                actual_asr_chunk_end_seconds = min(total_audio_duration_seconds, current_processing_window_start_seconds + CHUNK_LENGTH)
                
                start_sample_idx = int(actual_asr_chunk_start_seconds * MODEL_SAMPLE_RATE)
                end_sample_idx = int(actual_asr_chunk_end_seconds * MODEL_SAMPLE_RATE)

                if start_sample_idx >= end_sample_idx: # Should only happen if window is past audio end
                    break 
                
                # Extract 1D tensor chunk for ASR
                audio_chunk_for_asr = full_waveform[start_sample_idx:end_sample_idx] 
                
                if audio_chunk_for_asr.numel() == 0: # Skip if somehow the chunk is empty
                    current_processing_window_start_seconds += (CHUNK_LENGTH - OVERLAP)
                    continue # Advance window and re-evaluate loop condition

                # Call the ASR function for this chunk
                # `run_asr_on_tensor_chunk` expects a 1D tensor and its absolute start time
                segments_from_chunk, _ = await asyncio.to_thread( 
                    run_asr_on_tensor_chunk, 
                    audio_chunk_for_asr,      
                    actual_asr_chunk_start_seconds 
                )

                # Send processed segments to client, applying filtering logic
                if websocket.application_state == WebSocketState.CONNECTED:
                    for segment_data in segments_from_chunk:
                        # Filter: Send segment if its start time is within or after the current main window,
                        # or if it's the first segment overall (to capture audio at the very beginning).
                        if segment_data["start"] >= current_processing_window_start_seconds or not all_segments_sent_to_client:
                            segment_data["id"] = len(all_segments_sent_to_client) # Assign sequential ID
                            await websocket.send_json(segment_data)
                            all_segments_sent_to_client.append(segment_data)
                else:
                    logger.info("WebSocket (/ws): Client disconnected while sending segments.")
                    break 
                
                # Advance the main processing window
                current_processing_window_start_seconds += (CHUNK_LENGTH - OVERLAP)
            
            # 6. Send final transcription summary if still connected
            if websocket.application_state == WebSocketState.CONNECTED:
                final_transcription_text = " ".join(s['text'] for s in all_segments_sent_to_client).strip()
                transcription_duration = round(time.time() - processing_start_time, 3)
                await websocket.send_json({
                    "text": final_transcription_text, 
                    "language": "en", # Placeholder
                    "transcription_time": transcription_duration,
                    "total_segments": len(all_segments_sent_to_client),
                    "final_duration_processed_seconds": round(total_audio_duration_seconds, 3),
                    "type": "final_transcription"
                })
            logger.info("WebSocket (/ws): All processing complete.")

        except Exception as audio_processing_error:
            logger.error(f"WebSocket (/ws): Error during audio processing phase: {audio_processing_error}")
            import traceback; traceback.print_exc()
            if websocket.application_state == WebSocketState.CONNECTED:
                await websocket.send_json({"error": f"Audio processing error: {audio_processing_error}", "type": "error"})

    except Exception as outer_exception: # Catch any other unforeseen errors
        logger.error(f"WebSocket (/ws): Unhandled exception in handler: {str(outer_exception)}")
        import traceback; traceback.print_exc()
    finally:
        # Ensure WebSocket is closed from server-side if still open
        if websocket.application_state == WebSocketState.CONNECTED:
            await websocket.close()
        logger.info("WebSocket (/ws) handler finished.")


if __name__ == "__main__":
    import uvicorn
    if not asr_model:
        logger.critical("Cannot start server: ASR Model failed to load.")
    else:
        logger.info(f"Starting server on port {PORT}. Model Rate: {MODEL_SAMPLE_RATE}")
        uvicorn.run(app, host="0.0.0.0", port=PORT)
