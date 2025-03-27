# Design

1. Very basic RAG to chunk and look for parts of the CV that align with USCIS guidelines.
2. LLM-based (`gpt-4o`) text segmentation for better handling of CV/resume formats.
3. Zero-shot classification using `bart-large-mnli` for evidence categorization across 8 USCIS criteria.