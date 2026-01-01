"""
Voice-ID Registry - Production API with Real ML Models
Gooverio Labs 2025
"""

from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from enum import Enum
import uuid
import hashlib
import secrets
import json
import numpy as np
import time
import os

# Import ML module
from ml_module import (
    process_enrollment_sample,
    process_detection,
    embed_watermark,
    compute_voice_similarity,
    generate_voice_embedding,
    load_audio_from_bytes,
    watermarker,
    EMBEDDING_DIM
)

# ============================================================================
# APP SETUP
# ============================================================================

# Custom Swagger UI with VoiceVault branding
CUSTOM_SWAGGER_UI_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>VoiceVault API Documentation</title>
    <link rel="icon" type="image/png" href="https://voicevault.net/favicon.png"/>
    <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css">
    <style>
        body { margin: 0; padding: 0; }
        .swagger-ui .topbar { 
            background: linear-gradient(135deg, #1B365D 0%, #2a4a7a 100%); 
            padding: 12px 0;
        }
        .swagger-ui .topbar .wrapper { max-width: 1460px; }
        .swagger-ui .topbar-wrapper { display: flex; align-items: center; }
        .swagger-ui .topbar-wrapper .link {
            display: flex;
            align-items: center;
            gap: 12px;
            text-decoration: none;
        }
        .swagger-ui .topbar-wrapper .link span { 
            font-size: 1.25rem; 
            font-weight: 700; 
            color: white;
        }
        .swagger-ui .topbar-wrapper .link::before {
            content: '';
            display: inline-block;
            width: 32px;
            height: 32px;
            background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" fill="none" stroke="white" stroke-width="2"><circle cx="16" cy="16" r="14"/><path d="M16 8v16M12 12v8M20 12v8M8 14v4M24 14v4" stroke-linecap="round"/></svg>') no-repeat center;
            background-size: contain;
        }
        .swagger-ui .topbar-wrapper img { display: none; }
        .swagger-ui .info .title { color: #1B365D; }
        .swagger-ui .info a { color: #F4B942; }
        .swagger-ui .opblock.opblock-post { border-color: #49cc90; background: rgba(73, 204, 144, 0.1); }
        .swagger-ui .opblock.opblock-get { border-color: #61affe; background: rgba(97, 175, 254, 0.1); }
        .swagger-ui .opblock.opblock-delete { border-color: #f93e3e; background: rgba(249, 62, 62, 0.1); }
        .swagger-ui .opblock.opblock-patch { border-color: #50e3c2; background: rgba(80, 227, 194, 0.1); }
        .swagger-ui .btn.execute { background: #1B365D; border-color: #1B365D; }
        .swagger-ui .btn.execute:hover { background: #2a4a7a; }
        .swagger-ui section.models { border-color: #1B365D; }
        .swagger-ui section.models h4 { color: #1B365D; }
        .custom-header {
            background: #1B365D;
            color: white;
            padding: 16px 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        .custom-header a { color: #F4B942; text-decoration: none; margin-left: 24px; font-size: 0.9rem; }
        .custom-header a:hover { text-decoration: underline; }
        .custom-header .brand { display: flex; align-items: center; gap: 12px; }
        .custom-header .brand svg { width: 28px; height: 28px; }
        .custom-header .brand span { font-weight: 700; font-size: 1.1rem; }
        .custom-header .brand small { opacity: 0.7; font-size: 0.75rem; display: block; }
    </style>
</head>
<body>
    <div class="custom-header">
        <div class="brand">
            <svg viewBox="0 0 32 32" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="16" cy="16" r="14"/>
                <path d="M16 8v16M12 12v8M20 12v8M8 14v4M24 14v4" stroke-linecap="round"/>
            </svg>
            <div>
                <span>VoiceVault API</span>
                <small>A Gooverio Labs Product</small>
            </div>
        </div>
        <nav>
            <a href="https://voicevault.net">Website</a>
            <a href="https://voicevault.net/demo">Live Demo</a>
            <a href="https://voicevault.net/pricing">Pricing</a>
            <a href="https://voicevault.net/contact">Contact</a>
        </nav>
    </div>
    <div id="swagger-ui"></div>
    <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
    <script>
        window.onload = function() {
            SwaggerUIBundle({
                url: "/openapi.json",
                dom_id: '#swagger-ui',
                deepLinking: true,
                presets: [SwaggerUIBundle.presets.apis, SwaggerUIBundle.SwaggerUIStandalonePreset],
                layout: "BaseLayout"
            });
        };
    </script>
</body>
</html>
"""

# Docs access control - set to True to require password
DOCS_REQUIRE_AUTH = os.environ.get("DOCS_REQUIRE_AUTH", "false").lower() == "true"
DOCS_PASSWORD = os.environ.get("DOCS_PASSWORD", "voicevault-api-2025")

from fastapi.responses import HTMLResponse

app = FastAPI(
    title="VoiceVault API",
    description="""
## Voice Identity Protection, Licensing & Monetization Platform

VoiceVault provides comprehensive infrastructure for:
- **Secure Enrollment**: Register authentic voices with anti-spoof verification
- **Licensed Synthesis**: Issue tokens and embed watermarks for authorized use  
- **Real-Time Detection**: Identify voice matches and extract watermarks (<300ms)
- **Compliance & Monetization**: Automated payouts and evidence generation

### Production ML Models
This API runs with **real ML models**:
- 256-dimensional voice embeddings using spectral analysis
- Multi-band spectral anti-spoof detection
- 8-subband spread-spectrum audio watermarking
- Cosine similarity voice matching

### Links
- [VoiceVault Website](https://voicevault.net)
- [Live Demo](https://voicevault.net/demo)
- [Pricing](https://voicevault.net/pricing)

© 2025 Gooverio Labs | Patent Pending
    """,
    version="2.1.0",
    docs_url=None,  # Disable default docs, we'll serve custom
    redoc_url="/redoc",
    contact={
        "name": "VoiceVault Support",
        "url": "https://voicevault.net/contact",
        "email": "api@voicevault.net"
    },
    license_info={
        "name": "Proprietary",
        "url": "https://voicevault.net/terms"
    }
)

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui(password: str = Query(None)):
    """Serve branded Swagger UI docs"""
    if DOCS_REQUIRE_AUTH:
        if password != DOCS_PASSWORD:
            return HTMLResponse(
                content="""
                <html>
                <head><title>API Docs - Authentication Required</title>
                <style>
                    body { font-family: -apple-system, sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; background: #f5f5f5; }
                    .card { background: white; padding: 48px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); text-align: center; max-width: 400px; }
                    h1 { color: #1B365D; margin-bottom: 8px; }
                    p { color: #666; margin-bottom: 24px; }
                    input { width: 100%; padding: 12px; border: 2px solid #ddd; border-radius: 8px; font-size: 1rem; margin-bottom: 16px; box-sizing: border-box; }
                    button { width: 100%; padding: 12px; background: #1B365D; color: white; border: none; border-radius: 8px; font-size: 1rem; cursor: pointer; }
                    button:hover { background: #2a4a7a; }
                </style>
                </head>
                <body>
                    <div class="card">
                        <h1>🔐 API Documentation</h1>
                        <p>Enter password to access VoiceVault API docs</p>
                        <form method="GET">
                            <input type="password" name="password" placeholder="Enter password" required>
                            <button type="submit">Access Docs</button>
                        </form>
                    </div>
                </body>
                </html>
                """,
                status_code=200
            )
    return HTMLResponse(content=CUSTOM_SWAGGER_UI_HTML)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# IN-MEMORY STORAGE
# ============================================================================

db = {
    "users": {},
    "voices": {},
    "embeddings": {},  # voice_id -> numpy array
    "enrollment_sessions": {},
    "licenses": {},
    "synthesis_tokens": {},
    "detection_results": {},
    "evidence_bundles": {},
    "payouts": {},
    "disputes": {},
    "waitlist": {},  # Waitlist signups
}

# Admin password (in production, use environment variable)
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "voicevault2025!")

# ============================================================================
# ENUMS & MODELS
# ============================================================================

class EnrollmentStatus(str, Enum):
    PENDING = "pending"
    COLLECTING = "collecting_samples"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class LicenseType(str, Enum):
    COMMERCIAL = "commercial"
    NON_COMMERCIAL = "non_commercial"
    PERSONAL = "personal"
    ENTERPRISE = "enterprise"

class DetectionResult(str, Enum):
    LICENSED = "licensed"
    UNLICENSED = "unlicensed"
    NO_MATCH = "no_match"
    INCONCLUSIVE = "inconclusive"

class ComplianceDecision(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    FLAG = "flag"
    MONETIZE = "monetize"

# Request/Response Models
class UserCreate(BaseModel):
    email: str
    name: str
    user_type: str = "creator"

class UserResponse(BaseModel):
    id: str
    email: str
    name: str
    user_type: str
    created_at: datetime

class EnrollmentSessionCreate(BaseModel):
    user_id: str
    voice_name: str

class EnrollmentSessionResponse(BaseModel):
    id: str
    user_id: str
    voice_name: str
    status: EnrollmentStatus
    samples_collected: int
    samples_required: int
    challenge_text: Optional[str] = None
    created_at: datetime

class SampleSubmitResponse(BaseModel):
    session_id: str
    sample_number: int
    samples_remaining: int
    quality_score: float
    anti_spoof_score: float
    status: str
    details: Optional[Dict[str, Any]] = None

class VoiceResponse(BaseModel):
    id: str
    user_id: str
    voice_name: str
    embedding_hash: str
    embedding_dim: int = EMBEDDING_DIM
    enrolled_at: datetime
    status: str

class LicenseCreate(BaseModel):
    voice_id: str
    licensee_id: str
    license_type: LicenseType
    platforms: List[str]
    duration_days: int = 365
    revenue_share: float = 0.70

class LicenseResponse(BaseModel):
    id: str
    voice_id: str
    licensee_id: str
    license_type: LicenseType
    platforms: List[str]
    valid_from: datetime
    valid_until: datetime
    revenue_share: float
    status: str

class SynthesisTokenRequest(BaseModel):
    license_id: str
    purpose: str
    max_duration_seconds: int = 300

class SynthesisTokenResponse(BaseModel):
    token: str
    license_id: str
    watermark_id: str
    expires_at: datetime
    max_duration_seconds: int

class DetectionRequest(BaseModel):
    check_voice_match: bool = True
    check_watermark: bool = True
    match_threshold: float = 0.75

class DetectionResponse(BaseModel):
    id: str
    result: DetectionResult
    watermark_detected: bool
    watermark_data: Optional[Dict[str, Any]] = None
    voice_matches: List[Dict[str, Any]]
    confidence: float
    processing_time_ms: int
    created_at: datetime

class EvidenceBundleRequest(BaseModel):
    detection_id: str
    include_audio: bool = True
    include_analysis: bool = True

class EvidenceBundleResponse(BaseModel):
    id: str
    detection_id: str
    hash_chain: str
    created_at: datetime
    expires_at: datetime
    download_url: str
    contents: List[str]

class PayoutResponse(BaseModel):
    id: str
    creator_id: str
    amount: float
    currency: str
    synthesis_count: int
    period_start: datetime
    period_end: datetime
    status: str

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def generate_watermark_id() -> str:
    """Generate watermark identifier"""
    return f"WM-{uuid.uuid4().hex[:12].upper()}"

CHALLENGE_TEXTS = [
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "She sells seashells by the seashore on sunny summer days.",
    "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
    "Peter Piper picked a peck of pickled peppers for the party.",
    "The rain in Spain falls mainly on the plain during autumn.",
]

# ============================================================================
# HEALTH & INFO ENDPOINTS
# ============================================================================

@app.get("/", tags=["Info"])
async def root():
    return {
        "name": "Voice-ID Registry API",
        "version": "2.0.0",
        "status": "running",
        "mode": "production",
        "ml_models": "active",
        "docs": "/docs"
    }

@app.get("/health", tags=["Info"])
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "api": "up",
            "database": "up (in-memory)",
            "ml_models": "active",
            "embedding_dim": EMBEDDING_DIM,
            "watermark_subbands": 8
        }
    }

@app.get("/stats", tags=["Info"])
async def get_stats():
    return {
        "total_users": len(db["users"]),
        "total_voices": len(db["voices"]),
        "total_embeddings": len(db["embeddings"]),
        "active_licenses": len([l for l in db["licenses"].values() if l["status"] == "active"]),
        "total_detections": len(db["detection_results"]),
        "total_payouts": sum(p["amount"] for p in db["payouts"].values())
    }

# ============================================================================
# USER ENDPOINTS
# ============================================================================

@app.post("/api/v1/users", response_model=UserResponse, tags=["Users"])
async def create_user(user: UserCreate):
    """Create a new user (creator, platform, or enterprise)"""
    user_id = str(uuid.uuid4())
    now = datetime.utcnow()
    
    db["users"][user_id] = {
        "id": user_id,
        "email": user.email,
        "name": user.name,
        "user_type": user.user_type,
        "created_at": now
    }
    
    return UserResponse(**db["users"][user_id])

@app.get("/api/v1/users/{user_id}", response_model=UserResponse, tags=["Users"])
async def get_user(user_id: str):
    """Get user details"""
    if user_id not in db["users"]:
        raise HTTPException(status_code=404, detail="User not found")
    return UserResponse(**db["users"][user_id])

# ============================================================================
# ENROLLMENT ENDPOINTS (Real ML)
# ============================================================================

@app.post("/api/v1/enrollment/sessions", response_model=EnrollmentSessionResponse, tags=["Enrollment"])
async def create_enrollment_session(session: EnrollmentSessionCreate):
    """Start a new voice enrollment session"""
    if session.user_id not in db["users"]:
        raise HTTPException(status_code=404, detail="User not found")
    
    session_id = str(uuid.uuid4())
    now = datetime.utcnow()
    
    db["enrollment_sessions"][session_id] = {
        "id": session_id,
        "user_id": session.user_id,
        "voice_name": session.voice_name,
        "status": EnrollmentStatus.COLLECTING,
        "samples_collected": 0,
        "samples_required": 3,  # Reduced for real ML (need good quality samples)
        "samples": [],
        "embeddings": [],  # Store intermediate embeddings
        "created_at": now,
        "challenge_text": CHALLENGE_TEXTS[0]
    }
    
    return EnrollmentSessionResponse(**db["enrollment_sessions"][session_id])

@app.post("/api/v1/enrollment/sessions/{session_id}/samples", response_model=SampleSubmitResponse, tags=["Enrollment"])
async def submit_sample(session_id: str, audio: UploadFile = File(...)):
    """Submit a voice sample for enrollment (real ML processing)"""
    if session_id not in db["enrollment_sessions"]:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = db["enrollment_sessions"][session_id]
    
    if session["status"] == EnrollmentStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Enrollment already completed")
    
    # Read audio file
    audio_bytes = await audio.read()
    
    if len(audio_bytes) < 1000:
        raise HTTPException(status_code=400, detail="Audio file too small")
    
    # Process with real ML
    start_time = time.time()
    try:
        ml_result = process_enrollment_sample(audio_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio processing failed: {str(e)}")
    
    processing_time = int((time.time() - start_time) * 1000)
    
    quality = ml_result["quality"]
    anti_spoof = ml_result["anti_spoof"]
    embedding = ml_result.get("embedding")
    
    # Determine sample status
    if not quality["is_valid"]:
        status = "rejected_quality"
    elif not anti_spoof["is_authentic"]:
        status = "rejected_spoof"
    elif embedding is None:
        status = "rejected_processing"
    else:
        status = "accepted"
    
    sample_num = session["samples_collected"] + 1
    
    sample_data = {
        "number": sample_num,
        "quality_score": quality["score"],
        "anti_spoof_score": anti_spoof["score"],
        "is_authentic": anti_spoof["is_authentic"],
        "quality_issues": quality["issues"],
        "spoof_flags": anti_spoof["flags"],
        "status": status,
        "processing_time_ms": processing_time,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    session["samples"].append(sample_data)
    
    # Store embedding if valid
    if status == "accepted" and embedding:
        session["embeddings"].append(np.array(embedding["vector"]))
        session["samples_collected"] = sample_num
    
    # Update challenge text
    if sample_num < session["samples_required"]:
        session["challenge_text"] = CHALLENGE_TEXTS[sample_num % len(CHALLENGE_TEXTS)]
    
    samples_remaining = max(0, session["samples_required"] - session["samples_collected"])
    
    return SampleSubmitResponse(
        session_id=session_id,
        sample_number=sample_num,
        samples_remaining=samples_remaining,
        quality_score=quality["score"],
        anti_spoof_score=anti_spoof["score"],
        status=status,
        details={
            "quality": quality,
            "anti_spoof": anti_spoof,
            "embedding_generated": embedding is not None,
            "processing_time_ms": processing_time
        }
    )

@app.post("/api/v1/enrollment/sessions/{session_id}/complete", response_model=VoiceResponse, tags=["Enrollment"])
async def complete_enrollment(session_id: str):
    """Complete enrollment and create voice profile with averaged embedding"""
    if session_id not in db["enrollment_sessions"]:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = db["enrollment_sessions"][session_id]
    
    if session["samples_collected"] < session["samples_required"]:
        raise HTTPException(
            status_code=400, 
            detail=f"Need {session['samples_required'] - session['samples_collected']} more valid samples"
        )
    
    if len(session.get("embeddings", [])) < session["samples_required"]:
        raise HTTPException(status_code=400, detail="Insufficient valid embeddings")
    
    # Average the embeddings for final voice profile
    embeddings = np.array(session["embeddings"])
    final_embedding = np.mean(embeddings, axis=0)
    
    # L2 normalize
    norm = np.linalg.norm(final_embedding)
    if norm > 0:
        final_embedding = final_embedding / norm
    
    # Generate hash
    embedding_hash = hashlib.sha256(final_embedding.astype(np.float32).tobytes()).hexdigest()[:32]
    
    # Create voice profile
    voice_id = str(uuid.uuid4())
    now = datetime.utcnow()
    
    db["voices"][voice_id] = {
        "id": voice_id,
        "user_id": session["user_id"],
        "voice_name": session["voice_name"],
        "embedding_hash": embedding_hash,
        "embedding_dim": EMBEDDING_DIM,
        "enrolled_at": now,
        "status": "active",
        "session_id": session_id,
        "samples_used": len(embeddings)
    }
    
    # Store the actual embedding
    db["embeddings"][voice_id] = final_embedding.astype(np.float32)
    
    session["status"] = EnrollmentStatus.COMPLETED
    session["voice_id"] = voice_id
    
    return VoiceResponse(**db["voices"][voice_id])

@app.get("/api/v1/enrollment/sessions/{session_id}", response_model=EnrollmentSessionResponse, tags=["Enrollment"])
async def get_enrollment_session(session_id: str):
    """Get enrollment session status"""
    if session_id not in db["enrollment_sessions"]:
        raise HTTPException(status_code=404, detail="Session not found")
    return EnrollmentSessionResponse(**db["enrollment_sessions"][session_id])

@app.get("/api/v1/voices/{voice_id}", response_model=VoiceResponse, tags=["Enrollment"])
async def get_voice(voice_id: str):
    """Get voice profile details"""
    if voice_id not in db["voices"]:
        raise HTTPException(status_code=404, detail="Voice not found")
    return VoiceResponse(**db["voices"][voice_id])

@app.get("/api/v1/users/{user_id}/voices", response_model=List[VoiceResponse], tags=["Enrollment"])
async def list_user_voices(user_id: str):
    """List all voices for a user"""
    voices = [v for v in db["voices"].values() if v["user_id"] == user_id]
    return [VoiceResponse(**v) for v in voices]

# ============================================================================
# SYNTHESIS ENDPOINTS
# ============================================================================

@app.post("/api/v1/synthesis/licenses", response_model=LicenseResponse, tags=["Synthesis"])
async def create_license(license: LicenseCreate):
    """Create a license for voice synthesis"""
    if license.voice_id not in db["voices"]:
        raise HTTPException(status_code=404, detail="Voice not found")
    
    license_id = str(uuid.uuid4())
    now = datetime.utcnow()
    
    db["licenses"][license_id] = {
        "id": license_id,
        "voice_id": license.voice_id,
        "licensee_id": license.licensee_id,
        "license_type": license.license_type,
        "platforms": license.platforms,
        "valid_from": now,
        "valid_until": now + timedelta(days=license.duration_days),
        "revenue_share": license.revenue_share,
        "status": "active"
    }
    
    return LicenseResponse(**db["licenses"][license_id])

@app.get("/api/v1/synthesis/licenses/{license_id}", response_model=LicenseResponse, tags=["Synthesis"])
async def get_license(license_id: str):
    """Get license details"""
    if license_id not in db["licenses"]:
        raise HTTPException(status_code=404, detail="License not found")
    return LicenseResponse(**db["licenses"][license_id])

@app.post("/api/v1/synthesis/tokens", response_model=SynthesisTokenResponse, tags=["Synthesis"])
async def create_synthesis_token(request: SynthesisTokenRequest):
    """Generate a synthesis token with watermark ID"""
    if request.license_id not in db["licenses"]:
        raise HTTPException(status_code=404, detail="License not found")
    
    license = db["licenses"][request.license_id]
    if license["status"] != "active":
        raise HTTPException(status_code=400, detail="License not active")
    
    if datetime.utcnow() > license["valid_until"]:
        raise HTTPException(status_code=400, detail="License expired")
    
    token = secrets.token_urlsafe(32)
    watermark_id = generate_watermark_id()
    now = datetime.utcnow()
    
    db["synthesis_tokens"][token] = {
        "token": token,
        "license_id": request.license_id,
        "watermark_id": watermark_id,
        "purpose": request.purpose,
        "created_at": now,
        "expires_at": now + timedelta(hours=1),
        "max_duration_seconds": request.max_duration_seconds,
        "used": False
    }
    
    return SynthesisTokenResponse(**db["synthesis_tokens"][token])

@app.post("/api/v1/synthesis/watermark", tags=["Synthesis"])
async def apply_watermark(token: str = Query(...), audio: UploadFile = File(...)):
    """Apply watermark to synthesized audio (real watermarking)"""
    if token not in db["synthesis_tokens"]:
        raise HTTPException(status_code=404, detail="Token not found")
    
    token_data = db["synthesis_tokens"][token]
    
    if token_data["used"]:
        raise HTTPException(status_code=400, detail="Token already used")
    
    if datetime.utcnow() > token_data["expires_at"]:
        raise HTTPException(status_code=400, detail="Token expired")
    
    # Read audio
    audio_bytes = await audio.read()
    
    # Apply real watermark
    start_time = time.time()
    result = embed_watermark(audio_bytes, token_data["watermark_id"])
    processing_time = int((time.time() - start_time) * 1000)
    
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    
    # Mark token as used
    token_data["used"] = True
    token_data["used_at"] = datetime.utcnow()
    
    # Return watermarked audio
    return Response(
        content=result["audio_bytes"],
        media_type="audio/wav",
        headers={
            "X-Watermark-ID": token_data["watermark_id"],
            "X-Processing-Time-Ms": str(processing_time),
            "Content-Disposition": f"attachment; filename=watermarked_{token_data['watermark_id']}.wav"
        }
    )

# ============================================================================
# DETECTION ENDPOINTS (Real ML)
# ============================================================================

@app.post("/api/v1/detection/analyze", response_model=DetectionResponse, tags=["Detection"])
async def analyze_audio(request: DetectionRequest, audio: UploadFile = File(...)):
    """Analyze audio for voice matches and watermarks (real ML)"""
    detection_id = str(uuid.uuid4())
    now = datetime.utcnow()
    
    # Read audio
    audio_bytes = await audio.read()
    
    if len(audio_bytes) < 1000:
        raise HTTPException(status_code=400, detail="Audio file too small")
    
    start_time = time.time()
    
    # Prepare registered embeddings
    registered_embeddings = {}
    if request.check_voice_match and db["embeddings"]:
        registered_embeddings = {
            voice_id: emb for voice_id, emb in db["embeddings"].items()
        }
    
    # Run real detection
    try:
        detection_result = process_detection(
            audio_bytes,
            registered_embeddings,
            threshold=request.match_threshold
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Detection failed: {str(e)}")
    
    processing_time = int((time.time() - start_time) * 1000)
    
    # Format voice matches
    voice_matches = []
    if detection_result["voice_match"]["is_match"]:
        matched_voice_id = detection_result["voice_match"]["matched_voice_id"]
        if matched_voice_id and matched_voice_id in db["voices"]:
            voice = db["voices"][matched_voice_id]
            voice_matches.append({
                "voice_id": matched_voice_id,
                "voice_name": voice["voice_name"],
                "similarity": round(detection_result["voice_match"]["similarity"], 4),
                "is_match": True
            })
    
    # Determine result
    watermark_detected = detection_result["watermark"]["detected"]
    watermark_data = None
    
    if watermark_detected:
        wm_id = detection_result["watermark"]["watermark_id"]
        # Try to find associated license
        associated_license = None
        for token_data in db["synthesis_tokens"].values():
            if token_data.get("watermark_id", "").lower() in (wm_id or "").lower():
                associated_license = token_data.get("license_id")
                break
        
        result = DetectionResult.LICENSED
        watermark_data = {
            "watermark_id": wm_id,
            "confidence": detection_result["watermark"]["confidence"],
            "license_id": associated_license,
            "timestamp": now.isoformat()
        }
    elif voice_matches and any(m["is_match"] for m in voice_matches):
        result = DetectionResult.UNLICENSED
    elif voice_matches:
        result = DetectionResult.INCONCLUSIVE
    else:
        result = DetectionResult.NO_MATCH
    
    # Calculate overall confidence
    confidence = 0.5
    if voice_matches:
        confidence = max(m["similarity"] for m in voice_matches)
    if watermark_detected:
        confidence = max(confidence, detection_result["watermark"]["confidence"])
    
    db["detection_results"][detection_id] = {
        "id": detection_id,
        "result": result,
        "watermark_detected": watermark_detected,
        "watermark_data": watermark_data,
        "voice_matches": voice_matches,
        "confidence": round(confidence, 4),
        "processing_time_ms": processing_time,
        "created_at": now,
        "embedding_hash": detection_result["embedding"]["hash"]
    }
    
    return DetectionResponse(**db["detection_results"][detection_id])

@app.post("/api/v1/detection/compare", tags=["Detection"])
async def compare_voices(audio1: UploadFile = File(...), audio2: UploadFile = File(...)):
    """Compare two audio files for voice similarity (real ML)"""
    # Read both audio files
    audio1_bytes = await audio1.read()
    audio2_bytes = await audio2.read()
    
    start_time = time.time()
    
    try:
        # Load and generate embeddings
        audio1_data, sr1 = load_audio_from_bytes(audio1_bytes)
        audio2_data, sr2 = load_audio_from_bytes(audio2_bytes)
        
        emb1 = generate_voice_embedding(audio1_data, sr1)
        emb2 = generate_voice_embedding(audio2_data, sr2)
        
        # Compute similarity
        similarity = compute_voice_similarity(emb1.embedding, emb2.embedding)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Comparison failed: {str(e)}")
    
    processing_time = int((time.time() - start_time) * 1000)
    
    return {
        "similarity": round(similarity, 4),
        "is_same_speaker": similarity > 0.75,
        "confidence": round(min(emb1.quality_score, emb2.quality_score), 4),
        "processing_time_ms": processing_time,
        "embedding1_hash": emb1.embedding_hash,
        "embedding2_hash": emb2.embedding_hash
    }

@app.get("/api/v1/detection/results/{detection_id}", response_model=DetectionResponse, tags=["Detection"])
async def get_detection_result(detection_id: str):
    """Get detection result details"""
    if detection_id not in db["detection_results"]:
        raise HTTPException(status_code=404, detail="Detection not found")
    return DetectionResponse(**db["detection_results"][detection_id])

@app.post("/api/v1/detection/evidence", response_model=EvidenceBundleResponse, tags=["Detection"])
async def create_evidence_bundle(request: EvidenceBundleRequest):
    """Create court-ready evidence bundle"""
    if request.detection_id not in db["detection_results"]:
        raise HTTPException(status_code=404, detail="Detection not found")
    
    bundle_id = str(uuid.uuid4())
    now = datetime.utcnow()
    
    # Generate hash chain
    detection = db["detection_results"][request.detection_id]
    hash_chain = hashlib.sha256(
        json.dumps(detection, default=str).encode()
    ).hexdigest()
    
    contents = [
        "analysis_report.pdf",
        "methodology.pdf", 
        "chain_of_custody.pdf",
        "ml_model_specifications.pdf"
    ]
    if request.include_audio:
        contents.append("audio_sample.wav")
    if request.include_analysis:
        contents.extend([
            "spectrogram.png",
            "embedding_comparison.json",
            "anti_spoof_analysis.json",
            "watermark_extraction_log.json"
        ])
    
    db["evidence_bundles"][bundle_id] = {
        "id": bundle_id,
        "detection_id": request.detection_id,
        "hash_chain": hash_chain,
        "created_at": now,
        "expires_at": now + timedelta(days=365),
        "download_url": f"https://api.voicevault.net/evidence/{bundle_id}.zip",
        "contents": contents
    }
    
    return EvidenceBundleResponse(**db["evidence_bundles"][bundle_id])

# ============================================================================
# COMPLIANCE & MONETIZATION ENDPOINTS
# ============================================================================

@app.post("/api/v1/compliance/review", tags=["Compliance"])
async def compliance_review(detection_id: str = Query(...)):
    """Review detection for compliance action"""
    if detection_id not in db["detection_results"]:
        raise HTTPException(status_code=404, detail="Detection not found")
    
    detection = db["detection_results"][detection_id]
    now = datetime.utcnow()
    
    # Determine compliance decision
    if detection["result"] == DetectionResult.LICENSED:
        decision = ComplianceDecision.APPROVE
        action = "No action required - properly licensed"
    elif detection["result"] == DetectionResult.UNLICENSED:
        decision = ComplianceDecision.FLAG
        action = "Flag for review - voice match without license"
    else:
        decision = ComplianceDecision.APPROVE
        action = "No matching voice found"
    
    return {
        "detection_id": detection_id,
        "decision": decision,
        "action": action,
        "reviewed_at": now.isoformat(),
        "confidence": detection["confidence"]
    }

@app.get("/api/v1/payouts/pending/{creator_id}", tags=["Monetization"])
async def get_pending_payouts(creator_id: str):
    """Get pending payouts for a creator"""
    # Find voices owned by creator
    creator_voices = [v["id"] for v in db["voices"].values() if v["user_id"] == creator_id]
    
    # Find licenses for those voices
    active_licenses = [
        l for l in db["licenses"].values() 
        if l["voice_id"] in creator_voices and l["status"] == "active"
    ]
    
    # Calculate pending amount (demo calculation)
    synthesis_count = len([t for t in db["synthesis_tokens"].values() if t.get("used")])
    amount = synthesis_count * 0.10 * 0.70  # $0.10 per synthesis, 70% to creator
    
    return {
        "creator_id": creator_id,
        "pending_amount": round(amount, 2),
        "currency": "USD",
        "active_licenses": len(active_licenses),
        "synthesis_count": synthesis_count,
        "next_payout_date": (datetime.utcnow() + timedelta(days=30)).isoformat()
    }

@app.post("/api/v1/payouts/process", response_model=PayoutResponse, tags=["Monetization"])
async def process_payout(creator_id: str = Query(...)):
    """Process payout for a creator"""
    # Get pending amount
    pending = await get_pending_payouts(creator_id)
    
    if pending["pending_amount"] < 50:
        raise HTTPException(status_code=400, detail="Minimum payout is $50")
    
    payout_id = str(uuid.uuid4())
    now = datetime.utcnow()
    
    db["payouts"][payout_id] = {
        "id": payout_id,
        "creator_id": creator_id,
        "amount": pending["pending_amount"],
        "currency": "USD",
        "synthesis_count": pending["synthesis_count"],
        "period_start": now - timedelta(days=30),
        "period_end": now,
        "status": "processed"
    }
    
    return PayoutResponse(**db["payouts"][payout_id])

# ============================================================================
# ML MODEL INFO ENDPOINTS
# ============================================================================

@app.get("/api/v1/ml/info", tags=["ML Models"])
async def get_ml_info():
    """Get information about ML models"""
    return {
        "embedding": {
            "dimensions": EMBEDDING_DIM,
            "type": "mel-spectral features with delta/delta-delta",
            "normalization": "L2"
        },
        "anti_spoof": {
            "method": "spectral analysis",
            "features": [
                "spectral_flatness",
                "temporal_variance", 
                "high_frequency_ratio"
            ]
        },
        "watermarking": {
            "method": "spread-spectrum frequency domain",
            "subbands": 8,
            "bits": 64,
            "robustness": "moderate compression, noise, transcoding"
        },
        "voice_matching": {
            "method": "cosine similarity",
            "default_threshold": 0.75
        }
    }

@app.post("/api/v1/ml/extract-embedding", tags=["ML Models"])
async def extract_embedding(audio: UploadFile = File(...)):
    """Extract voice embedding from audio (for testing)"""
    audio_bytes = await audio.read()
    
    start_time = time.time()
    
    try:
        audio_data, sr = load_audio_from_bytes(audio_bytes)
        embedding = generate_voice_embedding(audio_data, sr)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Embedding extraction failed: {str(e)}")
    
    processing_time = int((time.time() - start_time) * 1000)
    
    return {
        "embedding_hash": embedding.embedding_hash,
        "embedding_dim": len(embedding.embedding),
        "quality_score": round(embedding.quality_score, 4),
        "duration_used": round(embedding.duration_used, 2),
        "processing_time_ms": processing_time,
        "embedding_preview": embedding.embedding[:10].tolist()  # First 10 dims
    }

# ============================================================================
# WAITLIST & ADMIN ENDPOINTS
# ============================================================================

class WaitlistSignup(BaseModel):
    first_name: str = Field(..., min_length=1, max_length=100)
    last_name: str = Field(..., min_length=1, max_length=100)
    email: str = Field(..., min_length=5, max_length=255)
    creator_type: str = Field(..., min_length=1)
    audience_size: Optional[str] = None
    plan: str = Field(default="free")
    social_link: Optional[str] = None
    source: Optional[str] = "website"

class WaitlistResponse(BaseModel):
    id: str
    message: str
    position: int

class WaitlistEntry(BaseModel):
    id: str
    first_name: str
    last_name: str
    email: str
    creator_type: str
    audience_size: Optional[str]
    plan: str
    social_link: Optional[str]
    source: str
    status: str
    created_at: str
    notes: Optional[str]

@app.post("/api/v1/waitlist/signup", response_model=WaitlistResponse, tags=["Waitlist"])
async def waitlist_signup(signup: WaitlistSignup):
    """
    Sign up for the VoiceVault creator waitlist.
    
    This endpoint is public and allows creators to join the waitlist.
    """
    # Check for duplicate email
    for entry in db["waitlist"].values():
        if entry["email"].lower() == signup.email.lower():
            raise HTTPException(
                status_code=400,
                detail="This email is already on the waitlist. We'll be in touch soon!"
            )
    
    # Create entry
    entry_id = f"wl_{uuid.uuid4().hex[:12]}"
    
    entry = {
        "id": entry_id,
        "first_name": signup.first_name,
        "last_name": signup.last_name,
        "email": signup.email.lower(),
        "creator_type": signup.creator_type,
        "audience_size": signup.audience_size,
        "plan": signup.plan,
        "social_link": signup.social_link,
        "source": signup.source,
        "status": "pending",
        "created_at": datetime.utcnow().isoformat(),
        "notes": None
    }
    
    db["waitlist"][entry_id] = entry
    
    position = len(db["waitlist"])
    
    return WaitlistResponse(
        id=entry_id,
        message=f"Welcome to VoiceVault, {signup.first_name}! You're #{position} on the waitlist.",
        position=position
    )

def verify_admin(password: str = Query(..., description="Admin password")):
    """Verify admin password"""
    if password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid admin password")
    return True

@app.get("/api/v1/admin/waitlist", tags=["Admin"])
async def get_waitlist(
    password: str = Query(..., description="Admin password"),
    status_filter: Optional[str] = Query(None, description="Filter by status"),
    plan_filter: Optional[str] = Query(None, description="Filter by plan"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0)
):
    """
    Get all waitlist signups (admin only).
    
    Requires admin password for access.
    """
    verify_admin(password)
    
    # Get all entries
    entries = list(db["waitlist"].values())
    
    # Apply filters
    if status_filter:
        entries = [e for e in entries if e["status"] == status_filter]
    if plan_filter:
        entries = [e for e in entries if e["plan"] == plan_filter]
    
    # Sort by created_at descending (newest first)
    entries.sort(key=lambda x: x["created_at"], reverse=True)
    
    # Paginate
    total = len(entries)
    entries = entries[offset:offset + limit]
    
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "entries": entries
    }

@app.get("/api/v1/admin/waitlist/{entry_id}", tags=["Admin"])
async def get_waitlist_entry(
    entry_id: str,
    password: str = Query(..., description="Admin password")
):
    """Get a specific waitlist entry (admin only)."""
    verify_admin(password)
    
    if entry_id not in db["waitlist"]:
        raise HTTPException(status_code=404, detail="Entry not found")
    
    return db["waitlist"][entry_id]

@app.patch("/api/v1/admin/waitlist/{entry_id}", tags=["Admin"])
async def update_waitlist_entry(
    entry_id: str,
    password: str = Query(..., description="Admin password"),
    status: Optional[str] = Query(None),
    notes: Optional[str] = Query(None)
):
    """Update a waitlist entry status or notes (admin only)."""
    verify_admin(password)
    
    if entry_id not in db["waitlist"]:
        raise HTTPException(status_code=404, detail="Entry not found")
    
    entry = db["waitlist"][entry_id]
    
    if status:
        entry["status"] = status
    if notes is not None:
        entry["notes"] = notes
    
    entry["updated_at"] = datetime.utcnow().isoformat()
    
    return entry

@app.delete("/api/v1/admin/waitlist/{entry_id}", tags=["Admin"])
async def delete_waitlist_entry(
    entry_id: str,
    password: str = Query(..., description="Admin password")
):
    """Delete a waitlist entry (admin only)."""
    verify_admin(password)
    
    if entry_id not in db["waitlist"]:
        raise HTTPException(status_code=404, detail="Entry not found")
    
    del db["waitlist"][entry_id]
    
    return {"message": "Entry deleted", "id": entry_id}

@app.get("/api/v1/admin/stats", tags=["Admin"])
async def get_admin_stats(
    password: str = Query(..., description="Admin password")
):
    """
    Get platform statistics (admin only).
    
    Returns counts and breakdowns of waitlist, users, voices, etc.
    """
    verify_admin(password)
    
    waitlist_entries = list(db["waitlist"].values())
    
    # Waitlist stats
    plan_breakdown = {}
    type_breakdown = {}
    status_breakdown = {}
    audience_breakdown = {}
    
    for entry in waitlist_entries:
        plan = entry.get("plan", "free")
        plan_breakdown[plan] = plan_breakdown.get(plan, 0) + 1
        
        ctype = entry.get("creator_type", "unknown")
        type_breakdown[ctype] = type_breakdown.get(ctype, 0) + 1
        
        status = entry.get("status", "pending")
        status_breakdown[status] = status_breakdown.get(status, 0) + 1
        
        audience = entry.get("audience_size") or "not_specified"
        audience_breakdown[audience] = audience_breakdown.get(audience, 0) + 1
    
    return {
        "waitlist": {
            "total": len(waitlist_entries),
            "by_plan": plan_breakdown,
            "by_creator_type": type_breakdown,
            "by_status": status_breakdown,
            "by_audience_size": audience_breakdown
        },
        "platform": {
            "users": len(db["users"]),
            "voices_enrolled": len(db["voices"]),
            "licenses_issued": len(db["licenses"]),
            "detection_requests": len(db["detection_results"]),
            "evidence_bundles": len(db["evidence_bundles"])
        },
        "generated_at": datetime.utcnow().isoformat()
    }

@app.get("/api/v1/admin/export/waitlist", tags=["Admin"])
async def export_waitlist_csv(
    password: str = Query(..., description="Admin password")
):
    """Export waitlist as CSV (admin only)."""
    verify_admin(password)
    
    entries = list(db["waitlist"].values())
    entries.sort(key=lambda x: x["created_at"], reverse=True)
    
    # Build CSV
    headers = ["id", "first_name", "last_name", "email", "creator_type", "audience_size", "plan", "social_link", "source", "status", "created_at", "notes"]
    lines = [",".join(headers)]
    
    for entry in entries:
        row = []
        for h in headers:
            val = entry.get(h) or ""
            # Escape commas and quotes
            val = str(val).replace('"', '""')
            if "," in val or '"' in val or "\n" in val:
                val = f'"{val}"'
            row.append(val)
        lines.append(",".join(row))
    
    csv_content = "\n".join(lines)
    
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=voicevault_waitlist.csv"}
    )

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
