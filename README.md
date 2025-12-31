# Voice-ID Registry API v2.0 - Real ML Models

Production API with real machine learning models for voice identity protection.

## 🧠 ML Models Included

### Voice Embedding (256-dimensional)
- Mel-frequency spectral analysis
- Delta and delta-delta features
- L2 normalized vectors
- Cosine similarity matching

### Anti-Spoof Detection
- Spectral flatness analysis
- Temporal variance detection
- High-frequency content ratio
- Multi-factor authenticity scoring

### Audio Watermarking
- Spread-spectrum frequency domain embedding
- 8 frequency subbands
- 64-bit watermark payload
- Robust to moderate compression/noise

### Voice Matching
- Cosine similarity computation
- Configurable matching threshold (default: 0.75)
- Confidence scoring

## 📦 Files

```
├── main.py           # FastAPI application
├── ml_module.py      # ML implementations
├── requirements.txt  # Python dependencies
├── Procfile         # Railway deployment
├── railway.json     # Railway config
├── runtime.txt      # Python version
└── .gitignore       # Git ignore rules
```

## 🚀 Deployment to Railway

### Option 1: Replace existing deployment
1. Delete all files in your `voiceid-api` GitHub repo
2. Upload these new files
3. Push to GitHub
4. Railway auto-deploys

### Option 2: Fresh deployment
1. Create new GitHub repo
2. Upload all files
3. Connect to Railway
4. Deploy

## 🔌 API Endpoints

### Health & Info
- `GET /` - API info
- `GET /health` - Health check
- `GET /stats` - Usage statistics
- `GET /docs` - Swagger UI

### Enrollment (Real ML)
- `POST /api/v1/enrollment/sessions` - Start enrollment
- `POST /api/v1/enrollment/sessions/{id}/samples` - Submit audio sample
- `POST /api/v1/enrollment/sessions/{id}/complete` - Complete enrollment

### Detection (Real ML)
- `POST /api/v1/detection/analyze` - Analyze audio for matches/watermarks
- `POST /api/v1/detection/compare` - Compare two audio files
- `POST /api/v1/detection/evidence` - Generate evidence bundle

### Synthesis
- `POST /api/v1/synthesis/licenses` - Create license
- `POST /api/v1/synthesis/tokens` - Generate synthesis token
- `POST /api/v1/synthesis/watermark` - Apply watermark to audio

### ML Info
- `GET /api/v1/ml/info` - ML model specifications
- `POST /api/v1/ml/extract-embedding` - Extract embedding (testing)

## 📋 Requirements

- Python 3.11+
- NumPy
- FastAPI
- Uvicorn

## 🧪 Testing

```bash
# Test ML module
python -c "from ml_module import *; print('ML module loaded!')"

# Run locally
uvicorn main:app --reload --port 8000
```

## 📝 Version History

- **v2.0.0** - Real ML models (current)
- **v1.0.0** - Simulated demo mode

---
© 2025 Gooverio Labs | Patent Pending
