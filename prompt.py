from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class PromptLoader:
    def __init__(self, prompt_dir: str):
        self.prompt_dir = Path(prompt_dir)

    def load_prompt(self, prompt_name: str) -> str:
        """Load prompt from file.
        
        Args:
            prompt_name: Name of the prompt file without extension
            
        Returns:
            The prompt text content
            
        Raises:
            FileNotFoundError: If prompt file doesn't exist
            IOError: If there's an error reading the file
        """
        try:
            prompt_path = self.prompt_dir / f"{prompt_name}.txt"
            if not prompt_path.exists():
                raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
                
            with open(prompt_path) as f:
                return f.read().strip()
        except Exception as e:
            logger.error(f"Error loading prompt {prompt_name}: {e}")
            raise
