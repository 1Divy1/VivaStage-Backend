PROMPT = """Transcript:
\"\"\"
{llm_input}
\"\"\"

Extract up to {max_reels} of the most interesting and engaging highlight moments.
Return fewer if the transcript naturally contains fewer high-quality moments.

Requirements:
- Highlights should ideally be 10–90 seconds long.
- Use natural content boundaries over artificial timing.
- Avoid overlapping timestamps.
- Reasons must be written in the same language as the transcript.
- Output must match the expected JSON schema."""