"""
Voice-ID Registry - Demo API
Gooverio Labs 2025
"""

from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from enum import Enum
import uuid
import hashlib
import secrets
import json

# ============================================================================
# APP SETUP
# ============================================================================

app = FastAPI(
    title="Voice-ID Registry API",
    description="""
## Voice Identity Protection, Licensing & Monetization Platform

Voice-ID Registry provides comprehensive infrastructure for:
- **Secure Enrollment**: Register authentic voices with anti-spoof verification
- **Licensed Synthesis**: Issue tokens and embed watermarks for authorized use
- **Real-Time Detection**: Identify voice matches and extract watermarks
- **Compliance & Monetization**: Automated payouts and evidence generation

### Demo Mode
This instance is running in demo mode with simulated ML models.
All core workflows are functional for demonstration purposes.

© 2025 Gooverio Labs
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# IN-MEMORY STORAGE (Demo)
# ============================================================================

db = {
    "users": {},
    "voices": {},
    "enrollment_sessions": {},
    "licenses": {},
    "synthesis_tokens": {},
    "detection_results": {},
    "evidence_bundles": {},
    "payouts": {},
    "disputes": {},
}

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
    voice_name: str = "My Voice"

class EnrollmentSessionResponse(BaseModel):
    id: str
    user_id: str
    voice_name: str
    status: EnrollmentStatus
    samples_collected: int
    samples_required: int
    created_at: datetime
    challenge_text: Optional[str] = None

class SampleSubmitResponse(BaseModel):
    session_id: str
    sample_number: int
    samples_remaining: int
    quality_score: float
    anti_spoof_score: float
    status: str

class VoiceResponse(BaseModel):
    id: str
    user_id: str
    voice_name: str
    embedding_hash: str
    enrolled_at: datetime
    status: str

class LicenseCreate(BaseModel):
    voice_id: str
    licensee_id: str
    license_type: LicenseType
    platforms: List[str] = ["all"]
    duration_days: int = 365
    revenue_share_creator: float = 0.70
    revenue_share_platform: float = 0.25

class LicenseResponse(BaseModel):
    id: str
    voice_id: str
    licensee_id: str
    license_type: LicenseType
    platforms: List[str]
    valid_from: datetime
    valid_until: datetime
    revenue_share_creator: float
    revenue_share_platform: float
    status: str

class SynthesisRequest(BaseModel):
    license_id: str
    text: str
    platform: str = "demo"

class SynthesisResponse(BaseModel):
    token_id: str
    license_id: str
    watermark_id: str
    audio_url: str
    duration_seconds: float
    created_at: datetime

class DetectionRequest(BaseModel):
    audio_url: Optional[str] = None
    check_watermark: bool = True
    check_voice_match: bool = True

class DetectionResponse(BaseModel):
    id: str
    result: DetectionResult
    watermark_detected: bool
    watermark_data: Optional[Dict] = None
    voice_matches: List[Dict]
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
# DEMO DATA GENERATION
# ============================================================================

def generate_embedding() -> str:
    """Simulate voice embedding generation"""
    return hashlib.sha256(secrets.token_bytes(32)).hexdigest()[:64]

def generate_watermark_id() -> str:
    """Generate watermark identifier"""
    return f"WM-{uuid.uuid4().hex[:12].upper()}"

def simulate_anti_spoof_score() -> float:
    """Simulate anti-spoof detection (demo always passes)"""
    return 0.95 + (secrets.randbelow(50) / 1000)

def simulate_quality_score() -> float:
    """Simulate audio quality assessment"""
    return 0.85 + (secrets.randbelow(150) / 1000)

def simulate_voice_similarity(is_match: bool = False) -> float:
    """Simulate voice similarity score"""
    if is_match:
        return 0.85 + (secrets.randbelow(150) / 1000)
    return 0.20 + (secrets.randbelow(300) / 1000)

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
        "version": "1.0.0",
        "status": "running",
        "mode": "demo",
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
            "ml_models": "simulated"
        }
    }

@app.get("/stats", tags=["Info"])
async def get_stats():
    return {
        "total_users": len(db["users"]),
        "total_voices": len(db["voices"]),
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
# ENROLLMENT ENDPOINTS
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
        "samples_required": 5,
        "samples": [],
        "created_at": now,
        "challenge_text": CHALLENGE_TEXTS[0]
    }
    
    return EnrollmentSessionResponse(**db["enrollment_sessions"][session_id])

@app.post("/api/v1/enrollment/sessions/{session_id}/samples", response_model=SampleSubmitResponse, tags=["Enrollment"])
async def submit_sample(session_id: str, audio: UploadFile = File(None)):
    """Submit a voice sample for enrollment"""
    if session_id not in db["enrollment_sessions"]:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = db["enrollment_sessions"][session_id]
    
    if session["status"] == EnrollmentStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Enrollment already completed")
    
    # Simulate sample processing
    quality_score = simulate_quality_score()
    anti_spoof_score = simulate_anti_spoof_score()
    
    sample_num = session["samples_collected"] + 1
    session["samples"].append({
        "number": sample_num,
        "quality": quality_score,
        "anti_spoof": anti_spoof_score,
        "timestamp": datetime.utcnow().isoformat()
    })
    session["samples_collected"] = sample_num
    
    # Update challenge text
    if sample_num < session["samples_required"]:
        session["challenge_text"] = CHALLENGE_TEXTS[sample_num % len(CHALLENGE_TEXTS)]
    
    samples_remaining = session["samples_required"] - sample_num
    
    return SampleSubmitResponse(
        session_id=session_id,
        sample_number=sample_num,
        samples_remaining=samples_remaining,
        quality_score=quality_score,
        anti_spoof_score=anti_spoof_score,
        status="accepted" if quality_score > 0.7 else "retry_recommended"
    )

@app.post("/api/v1/enrollment/sessions/{session_id}/complete", response_model=VoiceResponse, tags=["Enrollment"])
async def complete_enrollment(session_id: str):
    """Complete enrollment and create voice profile"""
    if session_id not in db["enrollment_sessions"]:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = db["enrollment_sessions"][session_id]
    
    if session["samples_collected"] < session["samples_required"]:
        raise HTTPException(
            status_code=400, 
            detail=f"Need {session['samples_required'] - session['samples_collected']} more samples"
        )
    
    # Create voice profile
    voice_id = str(uuid.uuid4())
    now = datetime.utcnow()
    embedding = generate_embedding()
    
    db["voices"][voice_id] = {
        "id": voice_id,
        "user_id": session["user_id"],
        "voice_name": session["voice_name"],
        "embedding_hash": embedding,
        "enrolled_at": now,
        "status": "active",
        "session_id": session_id
    }
    
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
        "revenue_share_creator": license.revenue_share_creator,
        "revenue_share_platform": license.revenue_share_platform,
        "status": "active"
    }
    
    return LicenseResponse(**db["licenses"][license_id])

@app.get("/api/v1/synthesis/licenses/{license_id}", response_model=LicenseResponse, tags=["Synthesis"])
async def get_license(license_id: str):
    """Get license details"""
    if license_id not in db["licenses"]:
        raise HTTPException(status_code=404, detail="License not found")
    return LicenseResponse(**db["licenses"][license_id])

@app.post("/api/v1/synthesis/synthesize", response_model=SynthesisResponse, tags=["Synthesis"])
async def synthesize_voice(request: SynthesisRequest):
    """Generate synthesized audio with watermark (simulated)"""
    if request.license_id not in db["licenses"]:
        raise HTTPException(status_code=404, detail="License not found")
    
    license = db["licenses"][request.license_id]
    
    if license["status"] != "active":
        raise HTTPException(status_code=400, detail="License is not active")
    
    if datetime.utcnow() > license["valid_until"]:
        raise HTTPException(status_code=400, detail="License has expired")
    
    if request.platform not in license["platforms"] and "all" not in license["platforms"]:
        raise HTTPException(status_code=403, detail="Platform not authorized")
    
    # Generate synthesis token
    token_id = str(uuid.uuid4())
    watermark_id = generate_watermark_id()
    now = datetime.utcnow()
    
    # Simulate audio duration based on text length
    duration = len(request.text.split()) * 0.4  # ~0.4s per word
    
    db["synthesis_tokens"][token_id] = {
        "token_id": token_id,
        "license_id": request.license_id,
        "voice_id": license["voice_id"],
        "watermark_id": watermark_id,
        "text": request.text,
        "platform": request.platform,
        "duration_seconds": duration,
        "created_at": now,
        "audio_url": f"https://demo.voiceid.registry/audio/{token_id}.wav"
    }
    
    return SynthesisResponse(
        token_id=token_id,
        license_id=request.license_id,
        watermark_id=watermark_id,
        audio_url=db["synthesis_tokens"][token_id]["audio_url"],
        duration_seconds=duration,
        created_at=now
    )

@app.get("/api/v1/synthesis/tokens/{token_id}", tags=["Synthesis"])
async def get_synthesis_token(token_id: str):
    """Get synthesis token details"""
    if token_id not in db["synthesis_tokens"]:
        raise HTTPException(status_code=404, detail="Token not found")
    return db["synthesis_tokens"][token_id]

# ============================================================================
# DETECTION ENDPOINTS
# ============================================================================

@app.post("/api/v1/detection/analyze", response_model=DetectionResponse, tags=["Detection"])
async def analyze_audio(request: DetectionRequest, audio: UploadFile = File(None)):
    """Analyze audio for voice matches and watermarks"""
    detection_id = str(uuid.uuid4())
    now = datetime.utcnow()
    
    # Simulate detection
    watermark_detected = secrets.randbelow(100) > 30  # 70% chance of watermark
    voice_matches = []
    
    # Simulate finding voice matches
    if db["voices"] and request.check_voice_match:
        for voice_id, voice in list(db["voices"].items())[:3]:
            is_match = secrets.randbelow(100) > 70  # 30% chance of match
            similarity = simulate_voice_similarity(is_match)
            if similarity > 0.5:
                voice_matches.append({
                    "voice_id": voice_id,
                    "voice_name": voice["voice_name"],
                    "similarity": round(similarity, 4),
                    "is_match": similarity > 0.80
                })
    
    # Determine result
    if watermark_detected:
        result = DetectionResult.LICENSED
        watermark_data = {
            "watermark_id": generate_watermark_id(),
            "license_id": list(db["licenses"].keys())[0] if db["licenses"] else None,
            "platform": "demo_platform",
            "timestamp": now.isoformat()
        }
    elif voice_matches and any(m["is_match"] for m in voice_matches):
        result = DetectionResult.UNLICENSED
        watermark_data = None
    elif voice_matches:
        result = DetectionResult.INCONCLUSIVE
        watermark_data = None
    else:
        result = DetectionResult.NO_MATCH
        watermark_data = None
    
    processing_time = 150 + secrets.randbelow(200)
    
    db["detection_results"][detection_id] = {
        "id": detection_id,
        "result": result,
        "watermark_detected": watermark_detected,
        "watermark_data": watermark_data,
        "voice_matches": voice_matches,
        "confidence": 0.85 + (secrets.randbelow(150) / 1000),
        "processing_time_ms": processing_time,
        "created_at": now
    }
    
    return DetectionResponse(**db["detection_results"][detection_id])

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
    
    contents = ["analysis_report.pdf", "methodology.pdf", "chain_of_custody.pdf"]
    if request.include_audio:
        contents.append("audio_sample.wav")
    if request.include_analysis:
        contents.extend(["spectrogram.png", "embedding_comparison.json"])
    
    db["evidence_bundles"][bundle_id] = {
        "id": bundle_id,
        "detection_id": request.detection_id,
        "hash_chain": hash_chain,
        "created_at": now,
        "expires_at": now + timedelta(days=365),
        "download_url": f"https://demo.voiceid.registry/evidence/{bundle_id}.zip",
        "contents": contents
    }
    
    return EvidenceBundleResponse(**db["evidence_bundles"][bundle_id])

# ============================================================================
# COMPLIANCE & MONETIZATION ENDPOINTS
# ============================================================================

@app.post("/api/v1/compliance/decisions", tags=["Compliance"])
async def process_compliance_decision(
    detection_id: str = Query(...),
    decision: ComplianceDecision = Query(...)
):
    """Process compliance decision for a detection"""
    if detection_id not in db["detection_results"]:
        raise HTTPException(status_code=404, detail="Detection not found")
    
    detection = db["detection_results"][detection_id]
    
    result = {
        "detection_id": detection_id,
        "decision": decision,
        "processed_at": datetime.utcnow().isoformat(),
        "actions_taken": []
    }
    
    if decision == ComplianceDecision.MONETIZE and detection["watermark_data"]:
        # Create payout record
        payout_id = str(uuid.uuid4())
        amount = 0.10  # Demo amount
        
        if detection["watermark_data"].get("license_id") in db["licenses"]:
            license = db["licenses"][detection["watermark_data"]["license_id"]]
            voice = db["voices"].get(license["voice_id"], {})
            
            db["payouts"][payout_id] = {
                "id": payout_id,
                "creator_id": voice.get("user_id", "unknown"),
                "amount": amount,
                "currency": "USD",
                "synthesis_count": 1,
                "period_start": datetime.utcnow(),
                "period_end": datetime.utcnow(),
                "status": "pending"
            }
            
            result["actions_taken"].append(f"Created payout {payout_id} for ${amount}")
    
    elif decision == ComplianceDecision.FLAG:
        result["actions_taken"].append("Flagged for manual review")
        result["actions_taken"].append("Notification sent to voice owner")
    
    elif decision == ComplianceDecision.REJECT:
        result["actions_taken"].append("Takedown notice prepared")
        result["actions_taken"].append("Evidence bundle auto-generated")
    
    return result

@app.get("/api/v1/compliance/payouts", response_model=List[PayoutResponse], tags=["Compliance"])
async def list_payouts(user_id: Optional[str] = None):
    """List payout records"""
    payouts = list(db["payouts"].values())
    if user_id:
        payouts = [p for p in payouts if p["creator_id"] == user_id]
    return [PayoutResponse(**p) for p in payouts]

@app.post("/api/v1/compliance/disputes", tags=["Compliance"])
async def create_dispute(detection_id: str = Query(...), reason: str = Query(...)):
    """Open a dispute for a detection result"""
    if detection_id not in db["detection_results"]:
        raise HTTPException(status_code=404, detail="Detection not found")
    
    dispute_id = str(uuid.uuid4())
    now = datetime.utcnow()
    
    db["disputes"][dispute_id] = {
        "id": dispute_id,
        "detection_id": detection_id,
        "reason": reason,
        "status": "open",
        "created_at": now,
        "updated_at": now
    }
    
    return {
        "dispute_id": dispute_id,
        "status": "open",
        "message": "Dispute created. Our team will review within 48 hours.",
        "created_at": now.isoformat()
    }

# ============================================================================
# DEMO HELPER ENDPOINTS
# ============================================================================

@app.post("/api/v1/demo/seed", tags=["Demo"])
async def seed_demo_data():
    """Seed database with demo data for presentations"""
    
    # Create demo creator
    creator_id = str(uuid.uuid4())
    db["users"][creator_id] = {
        "id": creator_id,
        "email": "alex.creator@example.com",
        "name": "Alex Creator",
        "user_type": "creator",
        "created_at": datetime.utcnow()
    }
    
    # Create demo platform
    platform_id = str(uuid.uuid4())
    db["users"][platform_id] = {
        "id": platform_id,
        "email": "partner@ttsplatform.com",
        "name": "TTS Platform Inc",
        "user_type": "platform",
        "created_at": datetime.utcnow()
    }
    
    # Create enrolled voice
    voice_id = str(uuid.uuid4())
    db["voices"][voice_id] = {
        "id": voice_id,
        "user_id": creator_id,
        "voice_name": "Alex's Voice",
        "embedding_hash": generate_embedding(),
        "enrolled_at": datetime.utcnow(),
        "status": "active"
    }
    
    # Create active license
    license_id = str(uuid.uuid4())
    now = datetime.utcnow()
    db["licenses"][license_id] = {
        "id": license_id,
        "voice_id": voice_id,
        "licensee_id": platform_id,
        "license_type": LicenseType.COMMERCIAL,
        "platforms": ["demo", "all"],
        "valid_from": now,
        "valid_until": now + timedelta(days=365),
        "revenue_share_creator": 0.70,
        "revenue_share_platform": 0.25,
        "status": "active"
    }
    
    return {
        "message": "Demo data seeded successfully",
        "data": {
            "creator_id": creator_id,
            "platform_id": platform_id,
            "voice_id": voice_id,
            "license_id": license_id
        }
    }

@app.delete("/api/v1/demo/reset", tags=["Demo"])
async def reset_demo():
    """Reset all demo data"""
    for key in db:
        db[key] = {}
    return {"message": "Demo data reset successfully"}

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
