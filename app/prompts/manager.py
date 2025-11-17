from pathlib import Path
from typing import Dict, Any, Optional
import json
from abc import ABC, abstractmethod

from app.core.logging import get_logger

logger = get_logger(__name__)


class PromptTemplate(ABC):
    """Abstract base class for prompt templates."""

    @abstractmethod
    def format_prompt(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """Format system and user prompts according to template."""
        pass


class ChatMLTemplate(PromptTemplate):
    """ChatML template for local models like Qwen3."""

    def format_prompt(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """Format prompts using ChatML format."""
        schema = kwargs.get('schema')
        model_name = kwargs.get('model_name', 'Response')

        formatted_system = system_prompt
        if schema:
            formatted_system += f"""

You must respond with valid JSON that matches this exact schema:

Schema for {model_name}:
{json.dumps(schema, indent=2)}

CRITICAL REQUIREMENTS:
1. Response must be ONLY valid JSON - no markdown, explanations, or additional text
2. JSON must match the schema exactly
3. All required fields must be present
4. Field types must match the schema specification"""

        return f"""<|im_start|>system
{formatted_system}<|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant
"""


class OpenAITemplate(PromptTemplate):
    """OpenAI template for API-based models."""

    def format_prompt(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """Format prompts for OpenAI API (handled by OpenAI client)."""
        # For OpenAI, we return the prompts separately since the client handles formatting
        return {
            'system': system_prompt,
            'user': user_prompt
        }


class PromptManager:
    """Manages loading and formatting of prompts for different providers."""

    def __init__(self):
        self.prompts_dir = Path(__file__).parent
        self._cache = {}

        # Template registry
        self.templates = {
            'chatml': ChatMLTemplate(),
            'openai': OpenAITemplate()
        }

    def load_prompt(self, provider_type: str, prompt_type: str, prompt_name: str) -> str:
        """
        Load a prompt from the organized directory structure.

        Args:
            provider_type: 'local' or 'api'
            prompt_type: 'system', 'user', or 'templates'
            prompt_name: name of the prompt file (without extension)

        Returns:
            Prompt content as string
        """
        cache_key = f"{provider_type}_{prompt_type}_{prompt_name}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        # Support both .py and .txt files
        for ext in ['.py', '.txt']:
            file_path = self.prompts_dir / provider_type / prompt_type / f"{prompt_name}{ext}"
            if file_path.exists():
                if ext == '.py':
                    # For Python files, expect a PROMPT variable
                    content = self._load_python_prompt(file_path)
                else:
                    # For text files, read directly
                    content = file_path.read_text(encoding='utf-8').strip()

                self._cache[cache_key] = content
                logger.debug(f"Loaded prompt: {cache_key}")
                return content

        raise FileNotFoundError(f"Prompt not found: {provider_type}/{prompt_type}/{prompt_name}")

    def _load_python_prompt(self, file_path: Path) -> str:
        """Load prompt from Python file."""
        import importlib.util

        spec = importlib.util.spec_from_file_location("prompt_module", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if hasattr(module, 'PROMPT'):
            return module.PROMPT
        elif hasattr(module, 'get_prompt'):
            return module.get_prompt()
        else:
            raise ValueError(f"Python prompt file must define PROMPT variable or get_prompt() function: {file_path}")

    def format_prompt(
        self,
        provider_type: str,
        system_prompt: str,
        user_prompt: str,
        template_name: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Format prompts using appropriate template for provider.

        Args:
            provider_type: 'local' or 'api'
            system_prompt: System prompt content
            user_prompt: User prompt content
            template_name: Optional specific template name
            **kwargs: Additional formatting parameters

        Returns:
            Formatted prompt string
        """
        # Auto-select template based on provider if not specified
        if template_name is None:
            template_name = 'chatml' if provider_type == 'local' else 'openai'

        if template_name not in self.templates:
            raise ValueError(f"Unknown template: {template_name}")

        template = self.templates[template_name]
        return template.format_prompt(system_prompt, user_prompt, **kwargs)

    def get_highlight_extraction_prompts(
        self,
        provider_type: str,
        number_of_reels: int,
        min_seconds: int,
        max_seconds: int,
        llm_input: str
    ) -> Dict[str, str]:
        """
        Get formatted system and user prompts for highlight extraction.

        Args:
            provider_type: 'local' or 'api'
            number_of_reels: Number of highlights to extract
            min_seconds: Minimum duration for highlights
            max_seconds: Maximum duration for highlights
            llm_input: Transcript content

        Returns:
            Dictionary with 'system' and 'user' prompts
        """
        # Load base prompts
        system_prompt = self.load_prompt(provider_type, 'system', 'highlight_extraction_system')
        user_prompt = self.load_prompt(provider_type, 'user', 'highlight_extraction_user')

        # Format with parameters
        formatted_system = system_prompt.format(
            min_seconds=min_seconds,
            max_seconds=max_seconds,
            number_of_reels=number_of_reels
        )

        formatted_user = user_prompt.format(
            llm_input=llm_input,
            number_of_reels=number_of_reels,
            min_seconds=min_seconds,
            max_seconds=max_seconds
        )

        return {
            'system': formatted_system,
            'user': formatted_user
        }

    def clear_cache(self):
        """Clear the prompt cache."""
        self._cache.clear()
        logger.debug("Prompt cache cleared")


# Global instance
prompt_manager = PromptManager()