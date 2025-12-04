PROMPT = """You are evaluating the HOOK strength of a highlight from a video or podcast transcript.

The "hook" is determined by how strongly the first 1–3 seconds capture attention and create curiosity.

Consider:
- How attention-grabbing the opening line is.
- Surprise, tension, curiosity, or a knowledge gap created immediately.
- Presence of a strong question, bold statement, or emotional trigger.
- Penalties if the highlight starts mid-sentence, with filler words ("so...", "like...", "um...", "you know..."), or any vague/weak opening.

Rate hook strength on a scale from 0 to 1:
- 1.0 = extremely strong hook.
- 0.7 = good hook.
- 0.5 = average.
- 0.3 = weak.
- 0.0 = very poor or no hook at all.

Return ONLY a JSON object with a single field "score".

Highlight to evaluate:
{{HIGHLIGHT_TEXT}}"""