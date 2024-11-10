import logging
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
import subprocess
import whisper
import torch
import torch.backends.mps
from pydub import AudioSegment

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class AudioTranscript:
    text: str
    segments: List[Dict[str, Any]]
    language: str

class AudioProcessor:
    def __init__(self, model_size: str = "medium"):
        """Initialize audio processor with specified Whisper model size."""
        try:
            # Device selection logic
            self.device = self._get_optimal_device()
            logger.info(f"Using device: {self.device}")
            
            self.model = whisper.load_model(model_size).to(self.device)
            logger.info(f"Successfully loaded Whisper model: {model_size}")
            
            # Check for ffmpeg installation
            try:
                subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
                self.has_ffmpeg = True
            except (subprocess.CalledProcessError, FileNotFoundError):
                self.has_ffmpeg = False
                logger.warning("FFmpeg not found. Please install ffmpeg for better audio extraction.")
                
        except Exception as e:
            logger.error(f"Error loading Whisper model: {e}")
            raise

    def _get_optimal_device(self) -> str:
        """
        Determine the optimal device for processing.
        Priority: CUDA > CPU > MPS (due to potential compatibility issues)
        """
        try:
            if torch.cuda.is_available():
                logger.info("CUDA GPU detected")
                return "cuda"
        except Exception as e:
            logger.warning(f"Error checking GPU availability: {e}")
        
        logger.info("Using CPU backend")
        return "cpu"

    def extract_audio(self, video_path: Path, output_dir: Path) -> Path:
        """Extract audio from video file and convert to format suitable for Whisper."""
        audio_path = output_dir / "audio.wav"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Extract audio using ffmpeg
            subprocess.run([
                "ffmpeg", "-i", str(video_path),
                "-vn",  # No video
                "-acodec", "pcm_s16le",  # PCM 16-bit little-endian
                "-ar", "16000",  # 16kHz sampling rate
                "-ac", "1",  # Mono
                "-y",  # Overwrite output
                str(audio_path)
            ], check=True, capture_output=True)
            
            logger.info("Successfully extracted audio using ffmpeg")
            return audio_path
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {e.stderr.decode()}")
            logger.info("Falling back to pydub for audio extraction...")
            
            try:
                video = AudioSegment.from_file(str(video_path))
                audio = video.set_channels(1).set_frame_rate(16000)
                audio.export(str(audio_path), format="wav")
                logger.info("Successfully extracted audio using pydub")
                return audio_path
            except Exception as e2:
                logger.error(f"Error extracting audio using pydub: {e2}")
                raise RuntimeError(
                    "Failed to extract audio. Please install ffmpeg using:\n"
                    "Ubuntu/Debian: sudo apt-get update && sudo apt-get install -y ffmpeg\n"
                    "MacOS: brew install ffmpeg\n"
                    "Windows: choco install ffmpeg"
                )

    def _detect_language(self, audio_path: Path) -> tuple[str, float]:
        """Detect language and confidence score of audio."""
        try:
            # Load and pad/trim the audio
            audio = whisper.load_audio(str(audio_path))
            if len(audio) > 30 * 16000:  # If longer than 30 seconds
                audio = audio[:30 * 16000]  # Use first 30 seconds
            
            # Run initial transcription with language detection
            result = self.model.transcribe(
                audio,
                task="translate",  # This forces language detection
                fp16=self.device == "cuda",
                language=None  # Let the model detect the language
            )
            
            language = result.get("language", "en")
            # Approximate confidence based on whether language was detected
            confidence = 0.8 if language != "en" else 0.5
            
            logger.info(f"Detected language: {language} with approximate confidence: {confidence:.2f}")
            return language, confidence
            
        except Exception as e:
            logger.error(f"Error detecting language: {e}")
            logger.exception(e)  # This will print the full traceback
            return "en", 0.0

    def _assess_transcription_quality(self, text: str) -> float:
        """Assess the quality of transcription based on various heuristics."""
        if not text:
            return 0.0
        
        # Basic quality checks
        words = text.split()
        if len(words) < 3:
            return 0.0
        
        # Check for common signs of poor transcription
        poor_quality_indicators = [
            len([w for w in words if len(w) > 20]),  # Very long "words"
            text.count('�'),  # Unicode errors
            text.count('['),  # Strange brackets
            text.count(']'),
            len([w for w in words if not any(c.isalnum() for c in w)])  # Non-alphanumeric words
        ]
        
        # Calculate quality score (0 to 1)
        base_score = 1.0
        for indicator in poor_quality_indicators:
            if indicator > 0:
                base_score *= 0.5
        
        return max(0.0, min(1.0, base_score))

    def transcribe(self, audio_path: Path) -> Optional[AudioTranscript]:
        """Transcribe audio file using Whisper with quality checks."""
        try:
            # First detect language
            language, lang_confidence = self._detect_language(audio_path)
            
            # Load audio
            audio = whisper.load_audio(str(audio_path))
            
            # If we're very confident it's not English, use that language
            transcription_options = {
                "task": "transcribe",
                "fp16": self.device == "cuda",
                "language": language if lang_confidence > 0.8 else None,
                "initial_prompt": "This is a transcription of spoken content.",
            }
            
            # Try first transcription
            logger.info("Starting initial transcription...")
            result = self.model.transcribe(audio, **transcription_options)
            quality_score = self._assess_transcription_quality(result["text"])
            
            # If quality is poor, try more aggressive approaches
            if quality_score < 0.5:
                logger.warning(f"Poor transcription quality ({quality_score:.2f}). Trying with different settings...")
                
                # Try with English forced
                transcription_options["language"] = "en"
                result = self.model.transcribe(audio, **transcription_options)
                new_quality_score = self._assess_transcription_quality(result["text"])
                
                # If still poor quality, try with larger segment size
                if new_quality_score < 0.5:
                    logger.warning("Still poor quality. Trying with larger segments...")
                    result = self.model.transcribe(
                        audio,
                        **transcription_options,
                        chunk_length=30,  # Longer chunks
                        best_of=5  # More candidates
                    )
                    quality_score = self._assess_transcription_quality(result["text"])
            
            # If we still have very poor quality, return None
            if quality_score < 0.2:
                logger.warning("Transcription quality too poor to use")
                return None
                
            return AudioTranscript(
                text=result["text"],
                segments=result["segments"],
                language=result["language"]
            )
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            logger.exception(e)  # This will print the full traceback
            return None