"""
Voice-ID Registry - ML Module
Real implementations for voice embedding, anti-spoof, watermarking, and detection.
Gooverio Labs 2025
"""

import numpy as np
import io
import struct
import hashlib
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

EMBEDDING_DIM = 256  # Voice embedding dimensions
SAMPLE_RATE = 16000  # Standard sample rate for voice
WATERMARK_SUBBANDS = 8  # Number of frequency subbands for watermarking
MIN_AUDIO_DURATION = 1.0  # Minimum audio duration in seconds
MAX_AUDIO_DURATION = 30.0  # Maximum audio duration in seconds

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class AudioAnalysis:
    """Results from audio quality analysis"""
    duration: float
    sample_rate: int
    rms_energy: float
    snr_estimate: float
    clipping_ratio: float
    silence_ratio: float
    quality_score: float
    is_valid: bool
    issues: List[str]

@dataclass 
class AntiSpoofResult:
    """Results from anti-spoof detection"""
    is_authentic: bool
    spoof_score: float  # 0 = definitely fake, 1 = definitely real
    spectral_flatness: float
    temporal_variance: float
    high_freq_ratio: float
    confidence: float
    flags: List[str]

@dataclass
class VoiceEmbedding:
    """Voice embedding result"""
    embedding: np.ndarray
    embedding_hash: str
    quality_score: float
    duration_used: float

@dataclass
class WatermarkResult:
    """Watermarking operation result"""
    success: bool
    watermark_id: str
    audio_data: Optional[bytes]
    strength: float
    message: str

@dataclass
class DetectionResult:
    """Voice detection/matching result"""
    is_match: bool
    similarity_score: float
    confidence: float
    matched_voice_id: Optional[str]
    watermark_detected: bool
    watermark_id: Optional[str]

# ============================================================================
# AUDIO UTILITIES
# ============================================================================

def load_audio_from_bytes(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
    """Load audio from bytes (WAV format expected)"""
    try:
        import wave
        
        # Try to parse as WAV
        with io.BytesIO(audio_bytes) as buf:
            with wave.open(buf, 'rb') as wav:
                sample_rate = wav.getframerate()
                n_channels = wav.getnchannels()
                sample_width = wav.getsampwidth()
                n_frames = wav.getnframes()
                
                raw_data = wav.readframes(n_frames)
                
                # Convert to numpy array
                if sample_width == 1:
                    dtype = np.uint8
                elif sample_width == 2:
                    dtype = np.int16
                elif sample_width == 4:
                    dtype = np.int32
                else:
                    raise ValueError(f"Unsupported sample width: {sample_width}")
                
                audio = np.frombuffer(raw_data, dtype=dtype).astype(np.float32)
                
                # Convert to mono if stereo
                if n_channels == 2:
                    audio = audio.reshape(-1, 2).mean(axis=1)
                
                # Normalize to [-1, 1]
                if sample_width == 1:
                    audio = (audio - 128) / 128.0
                elif sample_width == 2:
                    audio = audio / 32768.0
                elif sample_width == 4:
                    audio = audio / 2147483648.0
                
                return audio, sample_rate
                
    except Exception as e:
        logger.warning(f"WAV parsing failed: {e}, attempting raw float32")
        # Fallback: assume raw float32 at 16kHz
        audio = np.frombuffer(audio_bytes, dtype=np.float32)
        return audio, SAMPLE_RATE

def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Simple resampling using linear interpolation"""
    if orig_sr == target_sr:
        return audio
    
    duration = len(audio) / orig_sr
    target_length = int(duration * target_sr)
    
    # Linear interpolation
    x_old = np.linspace(0, 1, len(audio))
    x_new = np.linspace(0, 1, target_length)
    audio_resampled = np.interp(x_new, x_old, audio)
    
    return audio_resampled.astype(np.float32)

def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Normalize audio to [-1, 1] range"""
    max_val = np.abs(audio).max()
    if max_val > 0:
        return audio / max_val
    return audio

# ============================================================================
# AUDIO QUALITY ANALYSIS
# ============================================================================

def analyze_audio_quality(audio: np.ndarray, sample_rate: int) -> AudioAnalysis:
    """Comprehensive audio quality analysis"""
    issues = []
    
    # Duration check
    duration = len(audio) / sample_rate
    if duration < MIN_AUDIO_DURATION:
        issues.append(f"Audio too short: {duration:.2f}s < {MIN_AUDIO_DURATION}s")
    if duration > MAX_AUDIO_DURATION:
        issues.append(f"Audio too long: {duration:.2f}s > {MAX_AUDIO_DURATION}s")
    
    # RMS Energy (loudness)
    rms_energy = np.sqrt(np.mean(audio ** 2))
    if rms_energy < 0.01:
        issues.append("Audio level too low")
    
    # Clipping detection
    clipping_threshold = 0.99
    clipping_ratio = np.mean(np.abs(audio) > clipping_threshold)
    if clipping_ratio > 0.01:
        issues.append(f"Clipping detected: {clipping_ratio*100:.1f}%")
    
    # Silence detection
    silence_threshold = 0.02
    silence_ratio = np.mean(np.abs(audio) < silence_threshold)
    if silence_ratio > 0.5:
        issues.append(f"Too much silence: {silence_ratio*100:.1f}%")
    
    # Simple SNR estimation (using signal variance vs noise floor)
    # Split audio into frames
    frame_size = int(0.025 * sample_rate)  # 25ms frames
    hop_size = int(0.010 * sample_rate)    # 10ms hop
    
    frames = []
    for i in range(0, len(audio) - frame_size, hop_size):
        frames.append(audio[i:i+frame_size])
    
    if len(frames) > 0:
        frame_energies = [np.sqrt(np.mean(f ** 2)) for f in frames]
        signal_energy = np.percentile(frame_energies, 90)  # Top 10% as signal
        noise_energy = np.percentile(frame_energies, 10)   # Bottom 10% as noise
        
        if noise_energy > 0:
            snr_estimate = 20 * np.log10(signal_energy / noise_energy)
        else:
            snr_estimate = 60.0  # Very clean signal
    else:
        snr_estimate = 0.0
        issues.append("Audio too short for analysis")
    
    if snr_estimate < 10:
        issues.append(f"Low SNR: {snr_estimate:.1f} dB")
    
    # Calculate overall quality score
    quality_score = 1.0
    quality_score -= min(0.3, clipping_ratio * 10)  # Penalize clipping
    quality_score -= min(0.3, max(0, (0.5 - silence_ratio)) * -0.6)  # Penalize excess silence
    quality_score -= min(0.2, max(0, (20 - snr_estimate) / 100))  # Penalize low SNR
    quality_score -= min(0.1, max(0, (0.01 - rms_energy) * 10))  # Penalize low volume
    quality_score = max(0.0, min(1.0, quality_score))
    
    is_valid = len(issues) == 0 or (quality_score > 0.5 and duration >= MIN_AUDIO_DURATION)
    
    return AudioAnalysis(
        duration=duration,
        sample_rate=sample_rate,
        rms_energy=float(rms_energy),
        snr_estimate=float(snr_estimate),
        clipping_ratio=float(clipping_ratio),
        silence_ratio=float(silence_ratio),
        quality_score=float(quality_score),
        is_valid=is_valid,
        issues=issues
    )

# ============================================================================
# ANTI-SPOOF DETECTION
# ============================================================================

def detect_spoof(audio: np.ndarray, sample_rate: int) -> AntiSpoofResult:
    """
    Detect if audio is authentic human speech or synthetic/replayed.
    Uses spectral analysis techniques.
    """
    flags = []
    
    # 1. Spectral Flatness (Wiener entropy)
    # Real speech has more tonal structure, synthetic tends to be flatter
    fft_size = 2048
    hop_length = 512
    
    spectral_flatness_values = []
    for i in range(0, len(audio) - fft_size, hop_length):
        frame = audio[i:i+fft_size]
        spectrum = np.abs(np.fft.rfft(frame * np.hanning(fft_size)))
        spectrum = spectrum + 1e-10  # Avoid log(0)
        
        geometric_mean = np.exp(np.mean(np.log(spectrum)))
        arithmetic_mean = np.mean(spectrum)
        flatness = geometric_mean / arithmetic_mean if arithmetic_mean > 0 else 0
        spectral_flatness_values.append(flatness)
    
    spectral_flatness = np.mean(spectral_flatness_values) if spectral_flatness_values else 0.5
    
    # Synthetic audio often has very consistent flatness
    if spectral_flatness > 0.4:
        flags.append("High spectral flatness (possible synthetic)")
    
    # 2. Temporal variance
    # Real speech has natural variations, synthetic can be too consistent
    frame_energies = []
    frame_size = int(0.025 * sample_rate)
    for i in range(0, len(audio) - frame_size, frame_size):
        frame = audio[i:i+frame_size]
        frame_energies.append(np.sqrt(np.mean(frame ** 2)))
    
    temporal_variance = np.var(frame_energies) if len(frame_energies) > 1 else 0
    
    if temporal_variance < 0.001:
        flags.append("Low temporal variance (possible synthetic)")
    
    # 3. High frequency content ratio
    # Replayed/compressed audio often lacks high frequencies
    spectrum = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1/sample_rate)
    
    low_band = spectrum[freqs < 4000].sum()
    high_band = spectrum[(freqs >= 4000) & (freqs < 8000)].sum()
    
    high_freq_ratio = high_band / (low_band + 1e-10)
    
    if high_freq_ratio < 0.05:
        flags.append("Low high-frequency content (possible replay/compression)")
    
    # 4. Calculate spoof score
    # Higher score = more likely authentic
    spoof_score = 1.0
    
    # Penalize high spectral flatness
    spoof_score -= min(0.3, max(0, (spectral_flatness - 0.2) * 1.5))
    
    # Penalize low temporal variance  
    spoof_score -= min(0.3, max(0, (0.01 - temporal_variance) * 30))
    
    # Penalize low high-freq ratio
    spoof_score -= min(0.3, max(0, (0.1 - high_freq_ratio) * 3))
    
    spoof_score = max(0.0, min(1.0, spoof_score))
    
    # Determine confidence based on analysis consistency
    confidence = 0.7 + 0.3 * (1.0 - len(flags) / 3)
    
    is_authentic = spoof_score > 0.5
    
    return AntiSpoofResult(
        is_authentic=is_authentic,
        spoof_score=float(spoof_score),
        spectral_flatness=float(spectral_flatness),
        temporal_variance=float(temporal_variance),
        high_freq_ratio=float(high_freq_ratio),
        confidence=float(confidence),
        flags=flags
    )

# ============================================================================
# VOICE EMBEDDING GENERATION
# ============================================================================

def generate_voice_embedding(audio: np.ndarray, sample_rate: int) -> VoiceEmbedding:
    """
    Generate a voice embedding using spectral features.
    This is a simplified implementation that captures voice characteristics.
    For production, use Resemblyzer or ECAPA-TDNN.
    """
    # Resample to target rate
    if sample_rate != SAMPLE_RATE:
        audio = resample_audio(audio, sample_rate, SAMPLE_RATE)
        sample_rate = SAMPLE_RATE
    
    # Normalize
    audio = normalize_audio(audio)
    
    # Extract features using mel-frequency analysis
    # Parameters
    n_fft = 512
    hop_length = 160
    n_mels = 40
    
    # Compute STFT
    num_frames = 1 + (len(audio) - n_fft) // hop_length
    stft_matrix = np.zeros((n_fft // 2 + 1, num_frames), dtype=np.complex64)
    
    window = np.hanning(n_fft)
    for i in range(num_frames):
        start = i * hop_length
        frame = audio[start:start + n_fft]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        stft_matrix[:, i] = np.fft.rfft(frame * window)
    
    # Power spectrum
    power_spectrum = np.abs(stft_matrix) ** 2
    
    # Create mel filterbank
    mel_low = 0
    mel_high = 2595 * np.log10(1 + (sample_rate / 2) / 700)
    mel_points = np.linspace(mel_low, mel_high, n_mels + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    
    filterbank = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        for j in range(bin_points[i], bin_points[i + 1]):
            filterbank[i, j] = (j - bin_points[i]) / (bin_points[i + 1] - bin_points[i])
        for j in range(bin_points[i + 1], bin_points[i + 2]):
            filterbank[i, j] = (bin_points[i + 2] - j) / (bin_points[i + 2] - bin_points[i + 1])
    
    # Apply mel filterbank
    mel_spectrum = np.dot(filterbank, power_spectrum)
    mel_spectrum = np.log(mel_spectrum + 1e-10)
    
    # Extract statistics over time to create embedding
    embedding_parts = []
    
    # Mean of each mel band
    embedding_parts.append(np.mean(mel_spectrum, axis=1))  # 40 dims
    
    # Std of each mel band  
    embedding_parts.append(np.std(mel_spectrum, axis=1))   # 40 dims
    
    # Delta (first derivative) statistics
    delta = np.diff(mel_spectrum, axis=1)
    if delta.shape[1] > 0:
        embedding_parts.append(np.mean(delta, axis=1))     # 40 dims
        embedding_parts.append(np.std(delta, axis=1))      # 40 dims
    else:
        embedding_parts.append(np.zeros(n_mels))
        embedding_parts.append(np.zeros(n_mels))
    
    # Delta-delta (second derivative) statistics
    delta_delta = np.diff(delta, axis=1) if delta.shape[1] > 1 else np.zeros((n_mels, 1))
    if delta_delta.shape[1] > 0:
        embedding_parts.append(np.mean(delta_delta, axis=1))  # 40 dims
        embedding_parts.append(np.std(delta_delta, axis=1))   # 40 dims
    else:
        embedding_parts.append(np.zeros(n_mels))
        embedding_parts.append(np.zeros(n_mels))
    
    # Concatenate and resize to target dimension
    raw_embedding = np.concatenate(embedding_parts)  # 240 dims
    
    # Pad or truncate to target dimension
    if len(raw_embedding) < EMBEDDING_DIM:
        # Add spectral statistics to fill remaining dimensions
        spectral_centroid = np.sum(np.arange(mel_spectrum.shape[0])[:, np.newaxis] * mel_spectrum, axis=0) / (np.sum(mel_spectrum, axis=0) + 1e-10)
        extra_features = [
            np.mean(spectral_centroid),
            np.std(spectral_centroid),
            np.percentile(mel_spectrum.flatten(), 25),
            np.percentile(mel_spectrum.flatten(), 75),
        ]
        raw_embedding = np.concatenate([raw_embedding, extra_features])
    
    # Final resize
    if len(raw_embedding) > EMBEDDING_DIM:
        embedding = raw_embedding[:EMBEDDING_DIM]
    else:
        embedding = np.pad(raw_embedding, (0, EMBEDDING_DIM - len(raw_embedding)))
    
    # L2 normalize
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    
    # Generate hash for embedding
    embedding_bytes = embedding.astype(np.float32).tobytes()
    embedding_hash = hashlib.sha256(embedding_bytes).hexdigest()[:32]
    
    # Estimate quality based on frame count
    duration_used = len(audio) / sample_rate
    quality_score = min(1.0, duration_used / 5.0)  # Higher quality with more audio
    
    return VoiceEmbedding(
        embedding=embedding.astype(np.float32),
        embedding_hash=embedding_hash,
        quality_score=float(quality_score),
        duration_used=float(duration_used)
    )

# ============================================================================
# VOICE SIMILARITY / MATCHING
# ============================================================================

def compute_voice_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Compute cosine similarity between two voice embeddings.
    Returns value between 0 (different) and 1 (identical).
    """
    # Ensure embeddings are flattened
    e1 = embedding1.flatten()
    e2 = embedding2.flatten()
    
    # Cosine similarity
    dot_product = np.dot(e1, e2)
    norm1 = np.linalg.norm(e1)
    norm2 = np.linalg.norm(e2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    similarity = dot_product / (norm1 * norm2)
    
    # Clamp to [0, 1]
    return float(max(0.0, min(1.0, (similarity + 1) / 2)))

def match_voice(
    query_embedding: np.ndarray,
    registered_embeddings: Dict[str, np.ndarray],
    threshold: float = 0.75
) -> Tuple[Optional[str], float]:
    """
    Match a query embedding against registered voices.
    Returns (voice_id, similarity) of best match, or (None, 0) if no match.
    """
    best_match_id = None
    best_similarity = 0.0
    
    for voice_id, registered_embedding in registered_embeddings.items():
        similarity = compute_voice_similarity(query_embedding, registered_embedding)
        if similarity > best_similarity:
            best_similarity = similarity
            best_match_id = voice_id
    
    if best_similarity >= threshold:
        return best_match_id, best_similarity
    else:
        return None, best_similarity

# ============================================================================
# AUDIO WATERMARKING
# ============================================================================

class AudioWatermarker:
    """
    Audio watermarking using spread-spectrum technique in frequency domain.
    Embeds a 64-bit watermark across 8 frequency subbands.
    """
    
    def __init__(self, secret_key: str = "voicevault_default_key"):
        # Generate deterministic random sequence from key
        np.random.seed(int(hashlib.sha256(secret_key.encode()).hexdigest()[:8], 16))
        self.pn_sequence = np.random.randn(1024)  # Pseudo-noise sequence
        np.random.seed(None)  # Reset to true random
        
        self.frame_size = 2048
        self.hop_size = 512
        self.strength = 0.02  # Watermark strength (balance between robustness and imperceptibility)
    
    def _string_to_bits(self, s: str, num_bits: int = 64) -> np.ndarray:
        """Convert string to bit array"""
        # Hash the string to get fixed-length bit sequence
        hash_bytes = hashlib.sha256(s.encode()).digest()[:8]  # 64 bits
        bits = np.unpackbits(np.frombuffer(hash_bytes, dtype=np.uint8))
        return bits[:num_bits]
    
    def _bits_to_string(self, bits: np.ndarray) -> str:
        """Convert bit array back to hex string"""
        # Pack bits into bytes
        bits_padded = np.pad(bits, (0, 8 - len(bits) % 8 if len(bits) % 8 != 0 else 0))
        bytes_arr = np.packbits(bits_padded.astype(np.uint8))
        return bytes_arr.tobytes().hex()[:16]
    
    def embed(self, audio: np.ndarray, watermark_id: str, sample_rate: int) -> WatermarkResult:
        """Embed watermark into audio"""
        try:
            # Convert watermark ID to bits
            wm_bits = self._string_to_bits(watermark_id)
            
            # Ensure audio is float32
            audio = audio.astype(np.float32)
            watermarked = audio.copy()
            
            # Process audio in frames
            num_frames = (len(audio) - self.frame_size) // self.hop_size + 1
            
            if num_frames < len(wm_bits):
                return WatermarkResult(
                    success=False,
                    watermark_id=watermark_id,
                    audio_data=None,
                    strength=self.strength,
                    message="Audio too short for watermarking"
                )
            
            # Embed one bit per frame, cycling through watermark
            window = np.hanning(self.frame_size)
            
            for i in range(num_frames):
                bit_idx = i % len(wm_bits)
                bit = wm_bits[bit_idx]
                
                start = i * self.hop_size
                end = start + self.frame_size
                
                if end > len(audio):
                    break
                
                # Get frame and apply window
                frame = watermarked[start:end] * window
                
                # FFT
                spectrum = np.fft.rfft(frame)
                
                # Select subband for this bit (spread across frequency)
                subband_start = 50 + (bit_idx % WATERMARK_SUBBANDS) * 50
                subband_end = subband_start + 50
                
                # Modulate magnitude based on bit value
                pn_idx = (i * 7) % len(self.pn_sequence)
                pn_segment = self.pn_sequence[pn_idx:pn_idx + min(50, subband_end - subband_start)]
                
                if len(pn_segment) < subband_end - subband_start:
                    pn_segment = np.tile(pn_segment, 2)[:subband_end - subband_start]
                
                # Embed bit: 1 = add PN sequence, 0 = subtract
                sign = 1 if bit else -1
                spectrum_mag = np.abs(spectrum)
                spectrum_phase = np.angle(spectrum)
                
                # Modify magnitude in subband
                if subband_end <= len(spectrum_mag):
                    mod_factor = 1 + sign * self.strength * pn_segment[:subband_end-subband_start]
                    spectrum_mag[subband_start:subband_end] *= mod_factor
                
                # Reconstruct spectrum
                spectrum_modified = spectrum_mag * np.exp(1j * spectrum_phase)
                
                # IFFT
                frame_modified = np.fft.irfft(spectrum_modified, self.frame_size)
                
                # Overlap-add
                watermarked[start:end] += (frame_modified - frame) * window
            
            # Normalize to prevent clipping
            max_val = np.abs(watermarked).max()
            if max_val > 1.0:
                watermarked = watermarked / max_val * 0.99
            
            # Convert to bytes (WAV format)
            audio_bytes = self._audio_to_wav_bytes(watermarked, sample_rate)
            
            return WatermarkResult(
                success=True,
                watermark_id=watermark_id,
                audio_data=audio_bytes,
                strength=self.strength,
                message="Watermark embedded successfully"
            )
            
        except Exception as e:
            logger.error(f"Watermark embedding failed: {e}")
            return WatermarkResult(
                success=False,
                watermark_id=watermark_id,
                audio_data=None,
                strength=self.strength,
                message=f"Embedding failed: {str(e)}"
            )
    
    def extract(self, audio: np.ndarray, sample_rate: int) -> Tuple[bool, Optional[str], float]:
        """
        Extract watermark from audio.
        Returns (found, watermark_id, confidence)
        """
        try:
            audio = audio.astype(np.float32)
            
            num_frames = (len(audio) - self.frame_size) // self.hop_size + 1
            
            if num_frames < 64:
                return False, None, 0.0
            
            window = np.hanning(self.frame_size)
            
            # Extract bits from each frame
            extracted_bits = []
            bit_confidences = []
            
            for i in range(min(num_frames, 256)):  # Check up to 256 frames
                start = i * self.hop_size
                end = start + self.frame_size
                
                if end > len(audio):
                    break
                
                frame = audio[start:end] * window
                spectrum = np.fft.rfft(frame)
                spectrum_mag = np.abs(spectrum)
                
                bit_idx = i % 64
                subband_start = 50 + (bit_idx % WATERMARK_SUBBANDS) * 50
                subband_end = subband_start + 50
                
                # Correlate with PN sequence
                pn_idx = (i * 7) % len(self.pn_sequence)
                pn_segment = self.pn_sequence[pn_idx:pn_idx + min(50, subband_end - subband_start)]
                
                if len(pn_segment) < subband_end - subband_start:
                    pn_segment = np.tile(pn_segment, 2)[:subband_end - subband_start]
                
                if subband_end <= len(spectrum_mag):
                    subband = spectrum_mag[subband_start:subband_end]
                    correlation = np.corrcoef(subband[:len(pn_segment)], pn_segment)[0, 1]
                    
                    # Determine bit based on correlation sign
                    bit = 1 if correlation > 0 else 0
                    confidence = abs(correlation)
                    
                    extracted_bits.append(bit)
                    bit_confidences.append(confidence)
            
            if len(extracted_bits) < 64:
                return False, None, 0.0
            
            # Majority vote for each bit position
            bit_votes = np.zeros(64)
            bit_counts = np.zeros(64)
            
            for i, bit in enumerate(extracted_bits):
                pos = i % 64
                bit_votes[pos] += bit
                bit_counts[pos] += 1
            
            final_bits = (bit_votes / np.maximum(bit_counts, 1)) > 0.5
            avg_confidence = np.mean(bit_confidences)
            
            # Convert bits to watermark ID
            watermark_id = self._bits_to_string(final_bits.astype(np.int8))
            
            # Consider watermark found if confidence is above threshold
            found = avg_confidence > 0.1
            
            return found, watermark_id if found else None, float(avg_confidence)
            
        except Exception as e:
            logger.error(f"Watermark extraction failed: {e}")
            return False, None, 0.0
    
    def _audio_to_wav_bytes(self, audio: np.ndarray, sample_rate: int) -> bytes:
        """Convert audio array to WAV bytes"""
        import wave
        
        # Convert to int16
        audio_int16 = (audio * 32767).astype(np.int16)
        
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(audio_int16.tobytes())
        
        return buffer.getvalue()

# ============================================================================
# GLOBAL INSTANCES
# ============================================================================

watermarker = AudioWatermarker()

# ============================================================================
# HIGH-LEVEL API FUNCTIONS
# ============================================================================

def process_enrollment_sample(audio_bytes: bytes) -> Dict[str, Any]:
    """
    Process an audio sample for enrollment.
    Returns quality analysis, anti-spoof results, and embedding.
    """
    # Load audio
    audio, sample_rate = load_audio_from_bytes(audio_bytes)
    
    # Analyze quality
    quality = analyze_audio_quality(audio, sample_rate)
    
    # Anti-spoof check
    anti_spoof = detect_spoof(audio, sample_rate)
    
    # Generate embedding if audio is valid
    embedding = None
    if quality.is_valid and anti_spoof.is_authentic:
        embedding = generate_voice_embedding(audio, sample_rate)
    
    return {
        "quality": {
            "score": quality.quality_score,
            "duration": quality.duration,
            "snr": quality.snr_estimate,
            "is_valid": quality.is_valid,
            "issues": quality.issues
        },
        "anti_spoof": {
            "is_authentic": anti_spoof.is_authentic,
            "score": anti_spoof.spoof_score,
            "confidence": anti_spoof.confidence,
            "flags": anti_spoof.flags
        },
        "embedding": {
            "vector": embedding.embedding.tolist() if embedding else None,
            "hash": embedding.embedding_hash if embedding else None,
            "quality_score": embedding.quality_score if embedding else None
        } if embedding else None
    }

def process_detection(
    audio_bytes: bytes,
    registered_voices: Dict[str, np.ndarray],
    threshold: float = 0.75
) -> Dict[str, Any]:
    """
    Process audio for voice detection and watermark extraction.
    """
    # Load audio
    audio, sample_rate = load_audio_from_bytes(audio_bytes)
    
    # Generate embedding for query
    embedding = generate_voice_embedding(audio, sample_rate)
    
    # Match against registered voices
    matched_id, similarity = match_voice(
        embedding.embedding,
        registered_voices,
        threshold
    )
    
    # Extract watermark
    wm_found, wm_id, wm_confidence = watermarker.extract(audio, sample_rate)
    
    return {
        "voice_match": {
            "is_match": matched_id is not None,
            "matched_voice_id": matched_id,
            "similarity": similarity,
            "threshold": threshold
        },
        "watermark": {
            "detected": wm_found,
            "watermark_id": wm_id,
            "confidence": wm_confidence
        },
        "embedding": {
            "vector": embedding.embedding.tolist(),
            "hash": embedding.embedding_hash
        }
    }

def embed_watermark(audio_bytes: bytes, watermark_id: str) -> Dict[str, Any]:
    """
    Embed watermark into audio.
    """
    # Load audio
    audio, sample_rate = load_audio_from_bytes(audio_bytes)
    
    # Embed watermark
    result = watermarker.embed(audio, watermark_id, sample_rate)
    
    return {
        "success": result.success,
        "watermark_id": result.watermark_id,
        "audio_bytes": result.audio_data,
        "strength": result.strength,
        "message": result.message
    }
