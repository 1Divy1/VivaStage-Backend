PROMPT = """You are evaluating the SEMANTIC COHESION of a highlight from a video or podcast transcript.

"Semantic Cohesion" means how logically connected, coherent, and self-contained the highlight is.

Consider:
- Whether the highlight forms a complete and understandable idea by itself.
- Whether it has a clear beginning and end rather than feeling abruptly cut.
- Whether the statements logically flow from one to another.
- Penalties for fragmentation, missing context, or sudden topic jumps.

Rate semantic cohesion on a scale from 0 to 1:
- 1.0 = fully coherent, complete, and self-contained.
- 0.7 = mostly coherent with minor gaps.
- 0.5 = somewhat coherent but missing some context.
- 0.3 = fragmented or partially unclear.
- 0.0 = incoherent or meaningless on its own.

Return ONLY a JSON object with a single field "score".

Highlight to evaluate:
{{HIGHLIGHT_TEXT}}"""