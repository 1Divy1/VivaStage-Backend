PROMPT = """Transcript Chunk:
\"\"\"
{llm_input}
\"\"\"

You must identify exactly 2-5 high-quality key moments contained ONLY within the MAIN CHUNK section.

Evaluate potential moments using the following internal criteria:
- Semantic Importance
- Curiosity Trigger
- Content Density
- Clarity

Return ONLY key moments that meet these criteria.
Focus on substantial segments with complete thoughts, insights, or engaging discussions.
Ignore all low-quality, generic, fragmented, or incomplete segments.

STRICT REQUIREMENTS:
- Extract exactly 2-5 key moments (no more, no less)
- MANDATORY: Target 15-45 seconds per key moment for optimal engagement. Minimum 10 seconds, maximum 90 seconds.
- Prefer complete thoughts and concepts over sentence fragments
- Key moments must NOT overlap in time
- Use natural content boundaries where speakers complete their points
- Must be in chronological order
- Include timestamps with two decimals
- CRITICAL: Set reason field to empty string "" for ALL key moments - do NOT include any criteria names or explanations

Examples of good durations: 18s, 25s, 32s, 41s (substantial moments that tell a complete story).
Extract only the most meaningful, substantial moments from this chunk.
Output must match the required JSON schema.
"""