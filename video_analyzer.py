import argparse
from pathlib import Path
import json
from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass
import cv2
import shutil
import numpy as np
import subprocess
import tempfile
from clients.ollama import OllamaClient
from audio_processor import AudioProcessor, AudioTranscript
import torch
import torch.backends.mps

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Frame:
    number: int
    path: Path
    timestamp: float
    score: float

class Config:
    def __init__(self, config_path: str = "config/default_config.json"):
        self.config_path = Path(config_path)
        self.load_config()

    def load_config(self):
        """Load configuration from JSON file."""
        try:
            with open(self.config_path) as f:
                self.config = json.load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with optional default."""
        return self.config.get(key, default)

    def update_from_args(self, args: argparse.Namespace):
        """Update configuration with command line arguments."""
        for key, value in vars(args).items():
            if value is not None:  # Only update if argument was provided
                self.config[key] = value

class PromptLoader:
    def __init__(self, prompt_dir: str):
        self.prompt_dir = Path(prompt_dir)

    def load_prompt(self, prompt_name: str) -> str:
        """Load prompt from file."""
        try:
            with open(self.prompt_dir / f"{prompt_name}.txt") as f:
                return f.read().strip()
        except Exception as e:
            logger.error(f"Error loading prompt {prompt_name}: {e}")
            raise

class VideoProcessor:
    def __init__(self, video_path: Path, output_dir: Path, model: str):
        self.video_path = video_path
        self.output_dir = output_dir
        self.model = model
        self.frames: List[Frame] = []
        
    def _calculate_frame_difference(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate the difference between two frames using absolute difference."""
        if frame1 is None or frame2 is None:
            return 0.0
        
        # Convert to grayscale for simpler comparison
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference and mean
        diff = cv2.absdiff(gray1, gray2)
        score = np.mean(diff)
        
        return float(score)

    def _is_keyframe(self, current_frame: np.ndarray, prev_frame: np.ndarray, threshold: float = 10.0) -> bool:
        """Determine if frame is significantly different from previous frame."""
        if prev_frame is None:
            return True
            
        score = self._calculate_frame_difference(current_frame, prev_frame)
        return score > threshold

    def extract_keyframes(self, frames_per_minute: int = 10, duration: Optional[float] = None) -> List[Frame]:
        """Extract keyframes from video targeting a specific number of frames per minute."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps
        
        if duration:
            video_duration = min(duration, video_duration)
            total_frames = int(min(total_frames, duration * fps))
        
        # Calculate target number of frames
        target_frames = int((video_duration / 60) * frames_per_minute)
        target_frames = max(1, min(target_frames, total_frames))
        
        # Calculate adaptive sampling interval
        sample_interval = max(1, total_frames // (target_frames * 2))
        
        frame_candidates = []
        prev_frame = None
        frame_count = 0
        
        while frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % sample_interval == 0:
                if self._is_keyframe(frame, prev_frame):
                    score = self._calculate_frame_difference(frame, prev_frame)
                    frame_candidates.append((frame_count, frame, score))
                prev_frame = frame.copy()
                
            frame_count += 1
            
        cap.release()
        
        # Select the most significant frames
        frame_candidates.sort(key=lambda x: x[2], reverse=True)
        selected_candidates = frame_candidates[:target_frames]
        selected_candidates.sort(key=lambda x: x[0])  # Sort by frame number
        
        # Save selected frames
        self.frames = []
        for idx, (frame_num, frame, score) in enumerate(selected_candidates):
            frame_path = self.output_dir / f"frame_{idx}.jpg"
            cv2.imwrite(str(frame_path), frame)
            timestamp = frame_num / fps
            self.frames.append(Frame(idx, frame_path, timestamp, score))
        
        logger.info(f"Extracted {len(self.frames)} frames from video (target was {target_frames})")
        return self.frames

class VideoAnalyzer:
    def __init__(self, client: OllamaClient, model: str, prompt_loader: PromptLoader):
        self.client = client
        self.model = model
        self.prompt_loader = prompt_loader
        self._load_prompts()
        
    def _load_prompts(self):
        """Load prompts from files."""
        self.frame_prompt = self.prompt_loader.load_prompt("frame_analysis")
        self.video_prompt = self.prompt_loader.load_prompt("video_reconstruction")
        self.narrative_prompt = self.prompt_loader.load_prompt("narrate_storyteller")

    def analyze_frame(self, frame: Frame) -> Dict[str, Any]:
        """Analyze a single frame using the LLM."""
        prompt = f"{self.frame_prompt}\nThis is frame {frame.number} captured at {frame.timestamp:.2f} seconds."
        
        try:
            response = self.client.generate(
                prompt=prompt,
                image_path=str(frame.path),
                model=self.model,
                num_predict=300
            )
            logger.info(f"Successfully analyzed frame {frame.number}")
            return response
        except Exception as e:
            logger.error(f"Error analyzing frame {frame.number}: {e}")
            return {"response": f"Error analyzing frame {frame.number}: {str(e)}"}

    def reconstruct_video(self, frame_analyses: List[Dict[str, Any]], frames: List[Frame], 
                         transcript: Optional[AudioTranscript] = None) -> Dict[str, Any]:
        """Reconstruct video description from frame analyses and transcript."""
        frame_notes = []
        for i, (frame, analysis) in enumerate(zip(frames, frame_analyses)):
            frame_note = (
                f"Frame {i} ({frame.timestamp:.2f}s):\n"
                f"{analysis.get('response', 'No analysis available')}"
            )
            frame_notes.append(frame_note)
        
        analysis_text = "\n\n".join(frame_notes)
        
        # Include transcript information if available and of good quality
        transcript_text = ""
        if transcript and transcript.text.strip():
            transcript_text = (
                f"\nAudio Transcript:\n{transcript.text}\n\n"
                f"Detailed segments:\n"
            )
            for segment in transcript.segments:
                transcript_text += (
                    f"[{segment['start']:.1f}s - {segment['end']:.1f}s]: "
                    f"{segment['text']}\n"
                )
        
        prompt = (f"{self.video_prompt}\n\n"
                 f"Frame Notes:\n{analysis_text}\n\n"
                 f"{transcript_text}")
        
        try:
            response = self.client.generate(
                prompt=prompt,
                model=self.model,
                num_predict=1000
            )
            logger.info("Successfully reconstructed video description")
            return response
        except Exception as e:
            logger.error(f"Error reconstructing video: {e}")
            return {"response": f"Error reconstructing video: {str(e)}"}

    def enhance_narrative(self, technical_description: Dict[str, Any], 
                         transcript: Optional[AudioTranscript] = None) -> Dict[str, Any]:
        """Transform the technical video description into an engaging narrative."""
        tech_desc = technical_description.get('response', '')
        
        # Include transcript context if available
        transcript_context = ""
        if transcript and transcript.text.strip():
            transcript_context = f"\n\nSpoken content context:\n{transcript.text}\n"
        
        prompt = (f"{self.narrative_prompt}\n\n"
                 f"Here's the technical description to transform:\n\n"
                 f"{tech_desc}"
                 f"{transcript_context}")
        
        try:
            response = self.client.generate(
                prompt=prompt,
                model=self.model,
                num_predict=500
            )
            logger.info("Successfully enhanced video narrative")
            return response
        except Exception as e:
            logger.error(f"Error enhancing narrative: {e}")
            return {"response": f"Error enhancing narrative: {str(e)}"}

def cleanup_files(output_dir: Path):
    """Clean up temporary files and directories."""
    try:
        frames_dir = output_dir / "frames"
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
            logger.info(f"Cleaned up frames directory: {frames_dir}")
            
        audio_file = output_dir / "audio.wav"
        if audio_file.exists():
            audio_file.unlink()
            logger.info(f"Cleaned up audio file: {audio_file}")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

def main():
    parser = argparse.ArgumentParser(description="Analyze video using Ollama Vision model")
    parser.add_argument("video_path", type=str, help="Path to the video file")
    parser.add_argument("--config", type=str, default="config/default_config.json",
                        help="Path to configuration file")
    parser.add_argument("--output", type=str, help="Output directory for analysis results")
    parser.add_argument("--ollama-url", type=str, help="URL for the Ollama service")
    parser.add_argument("--model", type=str, help="Name of the vision model to use")
    parser.add_argument("--frames-per-minute", type=int, help="Target number of frames to extract per minute")
    parser.add_argument("--duration", type=float, help="Duration in seconds to process")
    parser.add_argument("--keep-frames", action="store_true", help="Keep extracted frames after analysis")
    parser.add_argument("--whisper-model", type=str, default="medium", 
                        help="Whisper model size (tiny, base, small, medium, large)")
    args = parser.parse_args()

    # Load and update configuration
    config = Config(args.config)
    config.update_from_args(args)

    # Initialize components
    video_path = Path(args.video_path)
    output_dir = Path(config.get("output_dir"))
    client = OllamaClient(config.get("ollama_url"))
    prompt_loader = PromptLoader(config.get("prompt_dir"))
    
    try:
        # Initialize audio processor and extract transcript
        logger.info("Initializing audio processing...")
        audio_processor = AudioProcessor(model_size=config.get("whisper_model"))
        
        logger.info("Extracting audio from video...")
        audio_path = audio_processor.extract_audio(video_path, output_dir)
        
        logger.info("Transcribing audio...")
        transcript = audio_processor.transcribe(audio_path)
        if transcript is None:
            logger.warning("Could not generate reliable transcript. Proceeding with video analysis only.")
        
        logger.info(f"Extracting frames from video using model {config.get('model')}...")
        processor = VideoProcessor(
            video_path, 
            output_dir / "frames", 
            config.get("model")
        )
        frames = processor.extract_keyframes(
            frames_per_minute=config.get("frames_per_minute"),
            duration=config.get("duration")
        )
        
        logger.info("Analyzing frames...")
        analyzer = VideoAnalyzer(client, config.get("model"), prompt_loader)
        frame_analyses = []
        for frame in frames:
            analysis = analyzer.analyze_frame(frame)
            frame_analyses.append(analysis)
            
        logger.info("Reconstructing video description...")
        technical_description = analyzer.reconstruct_video(
            frame_analyses, frames, transcript
        )

        logger.info("Enhancing narrative...")
        enhanced_narrative = analyzer.enhance_narrative(
            technical_description, transcript
        )
        
        output_dir.mkdir(parents=True, exist_ok=True)
        results = {
            "metadata": {
                "model_used": config.get("model"),
                "whisper_model": config.get("whisper_model"),
                "frames_per_minute": config.get("frames_per_minute"),
                "duration_processed": config.get("duration"),
                "frames_extracted": len(frames),
                "audio_language": transcript.language if transcript else None,
                "transcription_successful": transcript is not None
            },
            "transcript": {
                "text": transcript.text if transcript else None,
                "segments": transcript.segments if transcript else None
            } if transcript else None,
            "frame_analyses": frame_analyses,
            "technical_description": technical_description,
            "enhanced_narrative": enhanced_narrative
        }
        
        with open(output_dir / "analysis.json", "w") as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Analysis complete. Results saved to {output_dir / 'analysis.json'}")
        
        print("\nTranscript:")
        if transcript:
            print(transcript.text)
        else:
            print("No reliable transcript available")
        print("\nEnhanced Video Narrative:")
        print(enhanced_narrative.get("response", "No narrative generated"))
        
        if not config.get("keep_frames"):
            cleanup_files(output_dir)
            
    except Exception as e:
        logger.error(f"Error during video analysis: {e}")
        if not config.get("keep_frames"):
            cleanup_files(output_dir)
        raise

if __name__ == "__main__":
    main()