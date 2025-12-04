from pathlib import Path
import importlib.util

from app.providers.llm.base_llm_provider import LLMProvider
from app.pydantic_models.scoring.quality_score_model import QualityScoreModel
from app.core.logging import get_logger

logger = get_logger(__name__)


class ScoringEngine:
    """The component responsible for phase_2_scoring individual merged chunks

    Quality metrics: novelty score (40%), hook score (30%), emotional intensity (20%), semantic cohesion (10%)"""

    def __init__(self, chunk_content: str, llm_provider: LLMProvider):
        self.chunk_content = chunk_content
        self.llm_provider = llm_provider

    async def general_score(self) -> float:
        """Orchestrator method to calculate and return the weighted score"""
        try:
            # Get all individual scores (each between 0.0-1.0)
            novelty_score: float = await self._novelty_score()
            hook_score: float = await self._hook_score()
            emotional_intensity_score: float = await self._emotional_intensity_score()
            semantic_cohesion_score: float = await self._semantic_cohesion_score()

            # Apply weights: novelty (40%), hook (30%), emotional (20%), semantic (10%)
            weighted_score: float = (
                novelty_score * 0.4 +
                hook_score * 0.3 +
                emotional_intensity_score * 0.2 +
                semantic_cohesion_score * 0.1
            )

            logger.debug(f"Individual scores - Novelty: {novelty_score:.3f}, Hook: {hook_score:.3f}, "
                        f"Emotional: {emotional_intensity_score:.3f}, Semantic: {semantic_cohesion_score:.3f}")
            logger.debug(f"Final weighted score: {weighted_score:.3f}")

            return weighted_score

        except Exception as e:
            logger.error(f"Error calculating general score: {e}")
            return 0.5  # Fallback neutral score

    async def generate_explanation(self) -> str:
        """Generate detailed explanation for why this highlight was selected based on Phase 2 scoring criteria."""
        try:
            # Get all individual scores
            novelty_score: float = await self._novelty_score()
            hook_score: float = await self._hook_score()
            emotional_intensity_score: float = await self._emotional_intensity_score()
            semantic_cohesion_score: float = await self._semantic_cohesion_score()

            # Create explanation based on strongest qualities
            explanations = []

            if novelty_score > 0.7:
                explanations.append("presents novel ideas or unique insights")
            elif novelty_score > 0.5:
                explanations.append("offers fresh perspective on familiar topics")

            if hook_score > 0.7:
                explanations.append("creates strong engagement and captures attention")
            elif hook_score > 0.5:
                explanations.append("has engaging content that draws viewers in")

            if emotional_intensity_score > 0.7:
                explanations.append("delivers high emotional impact")
            elif emotional_intensity_score > 0.5:
                explanations.append("evokes meaningful emotional response")

            if semantic_cohesion_score > 0.7:
                explanations.append("maintains excellent thematic coherence")
            elif semantic_cohesion_score > 0.5:
                explanations.append("shows good topical consistency")

            if not explanations:
                explanations.append("meets baseline quality standards for content value")

            # Format as human-readable explanation
            explanation = f"Selected for Phase 2 scoring: {', '.join(explanations)}. Scores - Novelty: {novelty_score:.2f}, Hook: {hook_score:.2f}, Emotional: {emotional_intensity_score:.2f}, Semantic: {semantic_cohesion_score:.2f}"

            return explanation

        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return "Selected based on Phase 2 scoring analysis"

    async def _novelty_score(self) -> float:
        """Calculate novelty score using LLM evaluation."""
        try:
            system_prompt = self._load_scoring_prompt('scoring_system_prompt')
            user_prompt = self._load_scoring_prompt('novelty_scoring_prompt')

            # Replace placeholder with actual highlight content
            formatted_user_prompt = user_prompt.replace('{{HIGHLIGHT_TEXT}}', self.chunk_content)

            # Get LLM response
            response = await self.llm_provider.generate_structured_response(
                system_prompt=system_prompt,
                user_prompt=formatted_user_prompt,
                response_model=QualityScoreModel,
                model=None
            )

            return response.score

        except Exception as e:
            logger.error(f"Error calculating novelty score: {e}")
            return 0.5  # Fallback neutral score

    async def _hook_score(self) -> float:
        """Calculate hook strength score using LLM evaluation."""
        try:
            system_prompt = self._load_scoring_prompt('scoring_system_prompt')
            user_prompt = self._load_scoring_prompt('hook_scoring_prompt')

            # Replace placeholder with actual highlight content
            formatted_user_prompt = user_prompt.replace('{{HIGHLIGHT_TEXT}}', self.chunk_content)

            # Get LLM response
            response = await self.llm_provider.generate_structured_response(
                system_prompt=system_prompt,
                user_prompt=formatted_user_prompt,
                response_model=QualityScoreModel,
                model=None
            )

            return response.score

        except Exception as e:
            logger.error(f"Error calculating hook score: {e}")
            return 0.5  # Fallback neutral score

    async def _emotional_intensity_score(self) -> float:
        """Calculate emotional intensity score using LLM evaluation."""
        try:
            system_prompt = self._load_scoring_prompt('scoring_system_prompt')
            user_prompt = self._load_scoring_prompt('emotional_intensity_scoring_prompt')

            # Replace placeholder with actual highlight content
            formatted_user_prompt = user_prompt.replace('{{HIGHLIGHT_TEXT}}', self.chunk_content)

            # Get LLM response
            response = await self.llm_provider.generate_structured_response(
                system_prompt=system_prompt,
                user_prompt=formatted_user_prompt,
                response_model=QualityScoreModel,
                model=None
            )

            return response.score

        except Exception as e:
            logger.error(f"Error calculating emotional intensity score: {e}")
            return 0.5  # Fallback neutral score

    async def _semantic_cohesion_score(self) -> float:
        """Calculate semantic cohesion score using LLM evaluation."""
        try:
            system_prompt = self._load_scoring_prompt('scoring_system_prompt')
            user_prompt = self._load_scoring_prompt('semantic_cohesion_scoring_prompt')

            # Replace placeholder with actual highlight content
            formatted_user_prompt = user_prompt.replace('{{HIGHLIGHT_TEXT}}', self.chunk_content)

            # Get LLM response
            response = await self.llm_provider.generate_structured_response(
                system_prompt=system_prompt,
                user_prompt=formatted_user_prompt,
                response_model=QualityScoreModel,
                model=None
            )

            return response.score

        except Exception as e:
            logger.error(f"Error calculating semantic cohesion score: {e}")
            return 0.5  # Fallback neutral score

    def _load_scoring_prompt(self, prompt_name: str) -> str:
        """Load phase_2_scoring prompt from the prompts/phase_2_scoring directory."""
        prompts_dir = Path(__file__).parent.parent / 'prompts' / 'phase_2_scoring'
        file_path = prompts_dir / f'{prompt_name}.py'

        if not file_path.exists():
            raise FileNotFoundError(f"Scoring prompt not found: {file_path}")

        # Load the prompt from Python file
        spec = importlib.util.spec_from_file_location("prompt_module", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if hasattr(module, 'PROMPT'):
            return module.PROMPT
        else:
            raise ValueError(f"Prompt file must define PROMPT variable: {file_path}")