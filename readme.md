# Video Analysis using Llama3.2 Vision and OpenAI's Whisper Models locally

A video analysis tool that combines Llama's 11B vision model Whisper to create a description by taking key frames, feeding them to the vision model to get details. It uses the details from each frame and the transcript, if avaialable to describe what's happening in the video. 

It's designed to run 100% locally, but allow support openrouter.

## Features
- 💻 Runs completely locally - no cloud services or API keys needed
- 🎬 Intelligent key frame extraction from videos
- 🔊 High-quality audio transcription using OpenAI's Whisper
- 👁️ Frame analysis using Ollama and Llama3.2 11B Vision Model
- 📝 Natural language descriptions of video content
- 🔄 Automatic handling of poor quality audio
- 📊 Detailed JSON output of analysis results
- ⚙️ Highly configurable through command line arguments or config file

## Design
- Get key frames
- Get Audio
- Describe each frame independently with prompt template
- Compile all the responses and the transcription to describe the video
- Narrate the response for more natural consumption
![Description of the image](docs/design.png)

## Requirements

### System Requirements
- Python 3.11 or higher
- FFmpeg (required for audio processing)
- At least 16GB RAM (32GB recommended)
- GPU at least 12GB of VRAM or Apple M Series with at least 32GB

### Installation

1. Clone the repository:
```bash
git clone https://github.com/byjlw/video-analyzer.git
cd video-analyzer
```

2. Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install FFmpeg:
- Ubuntu/Debian:
  ```bash
  sudo apt-get update && sudo apt-get install -y ffmpeg
  ```
- macOS:
  ```bash
  brew install ffmpeg
  ```
- Windows:
  ```bash
  choco install ffmpeg
  ```

### Ollama Setup

1. Install Ollama following the instructions at [ollama.ai](https://ollama.ai)

2. Pull the default vision model:
```bash
ollama pull llama3.2-vision
```

3. Start the Ollama service:
```bash
ollama serve
```

### OpenRouter Setup (Optional)

If you want to use OpenRouter instead of Ollama:

1. Get an API key from [OpenRouter](https://openrouter.ai)
2. Either:
   - Pass it via command line: `--openrouter-key your-api-key`
   - Or add it to config/config.json:
     ```json
     {
       "clients": {
         "default": "openrouter",
         "openrouter": {
           "api_key": "your-api-key"
         }
       }
     }
     ```

## Project Structure

```
video-analyzer/
├── config/
│   └── default_config.json
├── prompts/
│   ├── frame_analysis.txt
│   ├── video_reconstruction.txt
│   └── narrate_storyteller.txt
├── output/             # Generated during runtime
├── video_analyzer.py   # Main script
└── requirements.txt
```

## Usage

### Basic Usage

Using Ollama (default):
```bash
python video_analyzer.py path/to/video.mp4
```

Using OpenRouter:
```bash
python video_analyzer.py path/to/video.mp4 --openrouter-key your-api-key
```

#### Sample Output
```
Transcript:
 Happy birthday to you!

Enhanced Video Narrative:
Here are the descriptions of what's happening in each frame:

**Frame 1 (2.50s)**
A woman enters a bedroom with a tray of breakfast items. She is wearing a white robe and holding a wooden tray, accompanied by a young girl with blonde hair who is smiling.

**To Frame 2 (4.50s)**
The woman moves to the right side of the image, still holding the tray of food and drinks. The young girl looks up at something outside the frame, while two other young girls with blue hoodies stand behind the woman, all smiling.

**To Frame 3 (5.00s)**
The woman holds the tray of food and drinks in front of the camera, with the two young girls with blue hoodies standing behind her. The food and drinks on the tray are assorted colors.

**To Frame 4 (5.50s)**
The woman moves slightly to the right and now holds the tray in front of a bunk bed. A blue curtain is visible behind the bunk bed, and she is wearing a white jacket.

**To Frame 5 (6.00s)**
The woman holds the tray in the center of the image, with a bed with a pink pillow visible against the wall to her right. She is now wearing a white shirt.

Note: The audio transcript indicates that "Happy Birthday to You!" is being sung, suggesting that this scene is taking place during a birthday celebration.
```

### Advanced Usage

```bash
python video_analyzer.py path/to/video.mp4 \
    --config custom_config.json \
    --output ./custom_output \
    --client openrouter \
    --openrouter-key your-api-key \
    --model llama3.2-vision \
    --frames-per-minute 15 \
    --duration 60 \
    --whisper-model medium \
    --keep-frames
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `video_path` | Path to the input video file | (Required) |
| `--config` | Path to configuration directory | config/ |
| `--output` | Output directory for analysis results | output/ |
| `--client` | Client to use (ollama or openrouter) | ollama |
| `--ollama-url` | URL for the Ollama service | http://localhost:11434 |
| `--openrouter-key` | API key for OpenRouter service | None |
| `--model` | Name of the vision model to use | llama3.2-vision |
| `--frames-per-minute` | Target number of frames to extract | 10 |
| `--duration` | Duration in seconds to process | None (full video) |
| `--whisper-model` | Whisper model size | medium |
| `--keep-frames` | Keep extracted frames after analysis | False |

## Configuration

The tool uses a cascading configuration system:
1. Command line arguments (highest priority)
2. User config (config/config.json)
3. Default config (config/default_config.json)

Example configuration:

```json
{
    "clients": {
        "default": "ollama",
        "ollama": {
            "url": "http://localhost:11434",
            "model": "llama3.2-vision"
        },
        "openrouter": {
            "api_key": "",
            "model": "llama3.2-vision"
        }
    },
    "prompt_dir": "prompts",
    "output_dir": "output",
    "frames": {
        "per_minute": 60,
        "analysis_threshold": 10.0,
        "min_difference": 5.0,
        "max_count": 30
    },
    "response_length": {
        "frame": 300,
        "reconstruction": 1000,
        "narrative": 500
    },
    "audio": {
        "whisper_model": "medium",
        "sample_rate": 16000,
        "channels": 1,
        "quality_threshold": 0.2,
        "chunk_length": 30,
        "language_confidence_threshold": 0.8
    },
    "keep_frames": false
}
```

### Configuration Options

#### General Settings
- `clients.default`: Default client to use (ollama or openrouter)
- `clients.ollama.url`: URL for the Ollama service
- `clients.ollama.model`: Vision model to use with Ollama
- `clients.openrouter.api_key`: API key for OpenRouter service
- `clients.openrouter.model`: Vision model to use with OpenRouter
- `prompt_dir`: Directory containing prompt files
- `output_dir`: Directory for output files
- `frames.per_minute`: Target number of frames to extract per minute
- `whisper_model`: Whisper model size (tiny, base, small, medium, large)
- `keep_frames`: Whether to keep extracted frames after analysis

#### Frame Analysis Settings
- `frames.analysis_threshold`: Threshold for key frame detection
- `frames.min_difference`: Minimum difference between frames
- `frames.max_count`: Maximum number of frames to extract

#### Response Length Settings
- `response_length.frame`: Maximum length for frame analysis
- `response_length.reconstruction`: Maximum length for video reconstruction
- `response_length.narrative`: Maximum length for enhanced narrative

#### Audio Settings
- `audio.sample_rate`: Audio sample rate
- `audio.channels`: Number of audio channels
- `audio.quality_threshold`: Minimum quality threshold for transcription
- `audio.chunk_length`: Length of audio chunks for processing
- `audio.language_confidence_threshold`: Confidence threshold for language detection

## Output

The tool generates a JSON file (`analysis.json`) containing:
- Metadata about the analysis
- Frame-by-frame analysis
- Audio transcript (if available)
- Technical description of the video
- Enhanced narrative description

### Example Output Structure

```json
{
    "metadata": {
        "client": "ollama",
        "model": "llama3.2-vision",
        "whisper_model": "medium",
        "frames_per_minute": 10,
        "frames_extracted": 15,
        "audio_language": "en",
        "transcription_successful": true
    },
    "transcript": {
        "text": "...",
        "segments": [...]
    },
    "frame_analyses": [...],
    "technical_description": {...},
    "enhanced_narrative": {...}
}
```

## Troubleshooting

### Common Issues

1. **FFmpeg not found**
   - Ensure FFmpeg is installed and accessible in your system PATH
   - Try reinstalling FFmpeg using the instructions above

2. **Ollama connection issues**
   - Verify Ollama is running: `ollama serve`
   - Check the URL in configuration matches your Ollama setup

3. **OpenRouter issues**
   - Verify your API key is correct
   - Check your OpenRouter account has sufficient credits

4. **Out of memory**
   - Try reducing `frames_per_minute`
   - Use a smaller Whisper model
   - Process shorter video duration using `--duration`

5. **Poor transcription quality**
   - Try using a larger Whisper model
   - Ensure good audio quality in the input video
   - Check for background noise in the video

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
