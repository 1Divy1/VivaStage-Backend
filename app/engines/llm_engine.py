from typing import List
from openai import OpenAI

from app.models.LLM.HighlightMoment import HighlightMoment
from app.models.LLM.HighlightMoments import HighlightMoments


class LLMEngine:

    def __init__(self, openai_client: OpenAI = None):
        self.openai_client = openai_client


    @staticmethod
    def create_llm_input_format(word_transcription: dict) -> str:
        """
        Create a formatted string for LLM input.
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


    def llm_inference(
        self,
        llm_model: str,
        system_prompt: str,
        user_prompt: str
    ) -> List[HighlightMoment]:
        """
        Get highlight moments from transcription using LLM.

        Args:
            llm_model (str): The model to use for inference.
            system_prompt (str): The system prompt to guide the LLM's behavior.
            user_prompt (str): The prompt to send to the LLM.

        Returns:
            dict: The response from the LLM containing highlight moments.
        """
        # TODO: Add a token to credit calculator
        response = self.openai_client.responses.parse(
            model=llm_model,
            instructions=system_prompt,  # System prompt
            input=user_prompt,           # User prompt
            text_format=HighlightMoments
        )
        structured_response: List[HighlightMoment] = response.output_parsed.highlights

        return structured_response

    def extract_highlights(self, llm_input: str, reel_data_input, llm_dir: str):
        """Extract highlight moments using LLM."""
        from app.core.logging import get_logger
        from app.utils.utils import save_data

        logger = get_logger(__name__)
        logger.info("Extracting highlights using LLM...")

        user_prompt = f"""
        Transcript:
        \"\"\"
        {llm_input}
        \"\"\"

        Please extract the first {reel_data_input.number_of_reels} most interesting and engaging highlight moments.

        - Each highlight must be **at least {reel_data_input.min_seconds} seconds**.
        - The preferred maximum is {reel_data_input.max_seconds} seconds, but it may be slightly exceeded (by up to 10s) if needed to preserve meaning.
        - Never return highlights below the minimum duration.
        """

        system_prompt = f"""
        You are a strict JSON generator and expert transcript analyst.

        Your task is to extract the most interesting and engaging highlight segments from a transcript of a video.

        You must obey the following rules:

        1. Each highlight must:
           - Be **at least {reel_data_input.min_seconds} seconds long**
           - **Ideally** be no longer than {reel_data_input.max_seconds} seconds, but you may go slightly over (up to ~10 seconds) if the highlight would otherwise be incomplete or incoherent.
           - Have a correctly calculated duration from `end - start` and never be shorter than the minimum under any condition.
           - Combine adjacent sentences if needed.
           - You must **skip** any segment that cannot satisfy the minimum time requirement.

        2. Do not explain your answer, do not include any commentary, markdown, or formatting.

        3. Format all times with two decimal places (e.g. "45.36").

        4. Return exactly {reel_data_input.number_of_reels} highlights (as requested by the user), unless there are not enough valid segments in the transcript.

        5. The "reason" field explaining why a segment is a highlight **must be written in the same language as the transcript text**.

        You are given the following fallback permissions:

        - You may cut off a sentence if doing so allows the segment to reach at least the minimum duration. When cutting off, make sure the segment ends naturally, without including a period (.), exclamation mark (!) or question mark (?) at the end.
        - Commas (,) at the end of the text are acceptable when a sentence is cut.

        You must strictly respect the minimum duration. Segments shorter than {reel_data_input.min_seconds} seconds are never allowed.
        """

        highlight_moments = self.llm_inference(
            llm_model="gpt-5-mini",
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )

        # Convert Pydantic models to dict for JSON serialization
        highlight_moments_dict = [moment.model_dump() for moment in highlight_moments]
        save_data(data=highlight_moments_dict, base_path=llm_dir, file_name="highlight_moments")

        return highlight_moments

