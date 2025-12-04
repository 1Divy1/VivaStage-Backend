PROMPT = """You are evaluating the EMOTIONAL INTENSITY of a highlight from a video or podcast transcript.

"Emotional Intensity" refers to the emotional impact conveyed through the content, such as:
- passion, excitement, surprise
- vulnerability, sadness, frustration
- humor, shock, boldness
- strong personal stories or powerful statements

Consider:
- How strongly the highlight evokes emotion or expresses an emotional tone.
- How much intensity or passion the speaker shows.
- How emotionally resonant or impactful the content is.

Rate emotional intensity on a scale from 0 to 1:
- 1.0 = extremely emotional or impactful.
- 0.7 = clearly emotional.
- 0.5 = moderate emotion.
- 0.3 = mild emotion.
- 0.0 = emotionally flat, neutral, or dry.

Return ONLY a JSON object with a single field "score".

Highlight to evaluate:
{{HIGHLIGHT_TEXT}}"""