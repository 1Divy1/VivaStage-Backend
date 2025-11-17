PROMPT = """Transcript:
\"\"\"
{llm_input}
\"\"\"

Please extract the first {number_of_reels} most interesting and engaging highlight moments.

- Each highlight must be **at least {min_seconds} seconds**.
- The preferred maximum is {max_seconds} seconds, but it may be slightly exceeded (by up to 10s) if needed to preserve meaning.
- Never return highlights below the minimum duration."""