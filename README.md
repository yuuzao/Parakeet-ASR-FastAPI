# Parakeet ASR

A high-performance Automatic Speech Recognition (ASR) server built with NVIDIA's NeMo Parakeet model. This application provides both REST API and WebSocket interfaces for transcribing audio files to text.

## 🚀 Features

- **State-of-the-art ASR**: Powered by NVIDIA's parakeet-tdt-0.6b-v2 model
- **Multiple APIs**:
  - REST API for simple file uploads
  - WebSocket API for real-time streaming transcription
- **Interactive Web UI**: Built-in browser interface for testing and demonstration
- **Docker Ready**: Easy deployment using Docker containers
- **Configurable**: Multiple environment variables to tune performance

## 📋 Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support (recommended)
- Docker (technically optional, but recommended for deployment)

## 🛠️ Installation

### Using Docker (Recommended)

The easiest way to run Parakeet ASR is using Docker:

```bash
# Clone the repository
git clone https://github.com/yourusername/parakeet-asr.git
cd parakeet-asr

# Build the Docker image
docker build -t parakeet-asr .

# Run the container
docker run --gpus all -p 8777:8777 parakeet-asr
```

### Manual Installation

If you prefer to run without Docker:

```bash
# Clone the repository
git clone https://github.com/yourusername/parakeet-asr.git
cd parakeet-asr

# Install dependencies
pip install -r app/requirements.txt

# Run the application
cd app
python main.py
```

## ⚙️ Configuration

The application can be configured using environment variables or a `.env` file in the app directory:

| Variable | Description | Default |
|----------|-------------|---------|
| `BATCH_SIZE` | ASR batch size during inference | 4 |
| `NUM_WORKERS` | Number of workers for processing | 0 |
| `TRANSCRIBE_CHUNK_LEN` | Audio chunk length in seconds | 30 |
| `TRANSCRIBE_OVERLAP` | Overlap between chunks in seconds | 5 |
| `SAMPLE_RATE` | Audio sample rate (Hz) | 16000 |
| `PORT` | Server port | 8777 |
| `LOG_LEVEL` | Logging level | INFO |

## 🔌 API Documentation

### REST API

#### Transcribe Audio

```
POST /v1/audio/transcriptions
```

**Request:**
- Content-Type: `multipart/form-data`
- Body: Form with an audio file attached as `file`

**Response:**
```json
{
  "text": "The complete transcribed text.",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 2.55,
      "text": "This is the first segment."
    },
    {
      "id": 1,
      "start": 2.56,
      "end": 5.2,
      "text": "This is the second segment."
    }
  ],
  "language": "en",
  "transcription_time": 1.234
}
```

### WebSocket API

Connect to `/v1/audio/transcriptions/ws` endpoint:

1. **Connection**: Connect to the WebSocket endpoint.
2. **Configuration**: Send a JSON message with audio configuration:
   ```json
   {
     "sample_rate": 16000,
     "channels": 1,
     "format": "binary"
   }
   ```
3. **Audio Data**: Send audio data in binary chunks.
4. **End Signal**: Send "END" as a text message to signal the end of the audio stream.
5. **Receiving Results**: The server sends JSON messages for each transcribed segment as they become available.
6. **Final Result**: After processing, a summary message with the full transcription is sent.

## 🖥️ Web Interface

A web interface is available at the root URL (`/`). This provides a simple way to test the transcription services:

- Upload audio files
- Choose between REST and WebSocket APIs
- View transcription results and timing information
- Debug mode for detailed logging

## 🛠️ Development

To set up a development environment:

```bash
# Clone the repository
git clone https://github.com/yourusername/parakeet-asr.git
cd parakeet-asr

# Install dev dependencies
pip install -r app/requirements.txt

# Run with debug logging
cd app
LOG_LEVEL=DEBUG python main.py
```

## 🔍 Troubleshooting

Common issues:

- **Model Loading Errors**: Ensure you have enough GPU memory available
- **Audio Format Issues**: The application works best with WAV files but supports various formats through torchaudio
- **Performance Issues**: Try adjusting `BATCH_SIZE`, `NUM_WORKERS`, and chunk settings in the `.env` file

## 📄 License

This project is licensed under the [MIT License](LICENSE).

## 🙏 Acknowledgements

- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) for the Parakeet ASR model
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [PyTorch](https://pytorch.org/) and [torchaudio](https://pytorch.org/audio) for audio processing
