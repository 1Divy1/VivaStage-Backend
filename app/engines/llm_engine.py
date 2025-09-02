from typing import List
from openai import OpenAI

from app.models.LLM.HighlightMoment import HighlightMoment
from app.models.LLM.HighlightMoments import HighlightMoments


class LLMEngine:

    def __init__(self, openai_client: OpenAI = None):
        self.openai_client = openai_client


    def create_llm_input_format(self, word_transcription: dict) -> str:
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

