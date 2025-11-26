from typing import List
import asyncio

from app.pydantic_models.shorts.short_model import ShortModel
from app.pydantic_models.shorts.shorts_response_model import ShortsResponseModel
from app.providers.llm.base_llm_provider import LLMProvider, LLMProviderError


class LLMEngine:

    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider


    @staticmethod
    def create_llm_input_format(word_transcription: dict) -> str:
        """
        Create a formatted string for shorts input.
        Raw words -> sentences (by punctuation) -> formatted sentences with timestamps.

        Args:
            word_transcription (dict): Dictionary containing words and their timestamps.

        Returns:
            str: Formatted sentences with timestamps.
        """
        sentence_endings = {".", "!", "?"}
        sentences = []
        current_sentence = []
        sentence_start_time = None

        # Iterate every word and create sentences from it
        for word in word_transcription["words"]:
            word_text = word["word"]
            start_time = word["start"]
            end_time = word["end"]

            # Reset sentence start time for a new sentence
            if sentence_start_time is None:
                sentence_start_time = start_time

            # Add the current word to the current sentence
            current_sentence.append({
                "word": word_text,
                "start": start_time,
                "end": end_time
            })

            # Check if a sentence is final
            if word_text[-1] in sentence_endings:
                sentence_text = " ".join(w["word"] for w in current_sentence)
                sentence_start_formatted = f"{sentence_start_time:.2f}"
                sentence_end_formatted = f"{current_sentence[-1]["end"]:.2f}"
                sentences.append({
                    "start": sentence_start_formatted,
                    "end": sentence_end_formatted,
                    "text": sentence_text
                })
                current_sentence = []
                sentence_start_time = None

        # If there are any sentences left
        if current_sentence:
            sentence_text = " ".join(w["word"] for w in current_sentence)
            sentence_start_formatted = f"{sentence_start_time:.2f}"
            sentence_end_formatted = f"{current_sentence[-1]["end"]:.2f}"
            sentences.append({
                "start": sentence_start_formatted,
                "end": sentence_end_formatted,
                "text": sentence_text
            })

        # Format sentences for output
        output: list[str] = []
        for s in sentences:
            line = f'[start: {s["start"]}, end: {s["end"]}] => "{s["text"]}"'
            output.append(line)

        formatted_output = "\n".join(output)
        return formatted_output


    async def llm_inference(
        self,
        llm_model: str,
        system_prompt: str,
        user_prompt: str
    ) -> List[ShortModel]:
        """
        Get highlight moments from transcription using shorts.

        Args:
            llm_model (str): The model to use for inference.
            system_prompt (str): The system prompt to guide the shorts's behavior.
            user_prompt (str): The prompt to send to the shorts.

        Returns:
            List[ShortModel]: The response from the shorts containing highlight moments.
        """
        try:
            # Use the abstracted provider for inference
            structured_response = await self.llm_provider.generate_structured_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=ShortsResponseModel,
                model=llm_model
            )

            return structured_response.highlights

        except LLMProviderError as e:
            from app.core.logging import get_logger
            logger = get_logger(__name__)
            logger.error(f"shorts inference failed: {e}")
            raise

    async def extract_highlights(self, llm_input: str, reel_data_input, llm_dir: str):
        """Extract highlight moments using shorts with organized prompt system."""
        from app.core.logging import get_logger
        from app.utils.utils import save_data
        from app.prompts.manager import prompt_manager

        logger = get_logger(__name__)
        logger.info("Extracting highlights using shorts...")

        # Determine provider type based on shorts provider
        provider_type = 'local' if self.llm_provider.provider_name == 'local' else 'api'

        # Get formatted prompts from prompt manager
        prompts = prompt_manager.get_highlight_extraction_prompts(
            provider_type=provider_type,
            number_of_reels=reel_data_input.number_of_reels,
            min_seconds=reel_data_input.min_seconds,
            max_seconds=reel_data_input.max_seconds,
            llm_input=llm_input
        )

        highlight_moments = await self.llm_inference(
            llm_model=None,  # Use default model from provider configuration
            system_prompt=prompts['system'],
            user_prompt=prompts['user']
        )

        # Convert Pydantic pydantic_models to dict for JSON serialization
        highlight_moments_dict = [moment.model_dump() for moment in highlight_moments]
        save_data(data=highlight_moments_dict, base_path=llm_dir, file_name="highlight_moments")

        return highlight_moments

