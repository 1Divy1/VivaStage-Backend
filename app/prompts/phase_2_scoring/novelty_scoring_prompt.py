PROMPT = """You are evaluating the NOVELTY of a highlight from a video or podcast transcript.

"Novelty" means:
- How new, insightful, surprising, or non-obvious the idea is.
- How much it goes beyond generic, cliché, or trivial statements.
- Whether it contains a memorable idea, perspective, or "aha moment".
- Whether it expresses a fresh insight, a unique angle, or an unexpected viewpoint.

Rate novelty on a scale from 0 to 1:
- 1.0 = extremely novel, insightful, or memorable.
- 0.7 = clearly interesting and somewhat fresh.
- 0.5 = somewhat interesting but partially generic.
- 0.3 = mostly generic or obvious.
- 0.0 = trivial or filler content.

Return ONLY a JSON object with a single field "score".

Highlight to evaluate:
{{HIGHLIGHT_TEXT}}"""