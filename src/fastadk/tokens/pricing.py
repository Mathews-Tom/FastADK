"""
pricing.py  —  Verified 2025-07-09

All prices are USD per 1 000 tokens: (input_price, output_price).

• Anthropic Cluade - https://docs.anthropic.com/en/docs/about-claude/pricing
• Gemini - https://ai.google.dev/gemini-api/docs/pricing
• OpenAI - https://openai.com/api/pricing/

Comments summarise headline capabilities or context size.

Prices divide the vendor-listed $/M by 1 000.  Cache/Bulk tiers not shown.
"""

DEFAULT_PRICING = {
    # ------------------------------------------------------------
    # OPENAI
    # ------------------------------------------------------------
    "openai": {
        # GPT-4.1 family – strongest general models
        "gpt-4.1": (0.002, 0.008),  # ⮕ SOTA reasoning & code, ≈1 M ctx
        "gpt-4.1-mini": (0.0004, 0.0016),  # ⮕  64 k+ ctx, fast & cheap
        "gpt-4.1-nano": (0.0001, 0.0004),  # ⮕  32 k ctx, ultra-low latency
        # GPT-4 family (legacy)
        "gpt-4": (0.03, 0.06),  # ⮕ Legacy GPT-4, used in tests
        # Reasoning-optimised “o” models
        "o3": (0.002, 0.008),  # ⮕ Top math / science / vision
        "o4-mini": (0.0011, 0.0044),  # ⮕ Faster reasoning, STEM-ready
        "o3-mini": (
            0.0011,
            0.0044,
        ),  # ⮕ Cost-efficient STEM reasoning; 200 k ctx, function calling, >o1-mini
        "o1": (0.015, 0.060),  # ⮕ Deep chain-of-thought, vision
        # Multimodal GPT-4o
        "gpt-4o": (0.005, 0.020),  # ⮕ Real-time text-vision-audio
        "gpt-4o-mini": (0.0006, 0.0024),  # ⮕  82 % MMLU at budget scale
        # Everyday chat
        "gpt-3.5-turbo-0125": (0.0005, 0.0015),  # ⮕ Value chat, 16 k ctx
        # Vector embeddings v3
        "text-embedding-3-small": (0.00002, 0.00002),  # ⮕ 512-d, multilingual
        "text-embedding-3-large": (0.00013, 0.00013),  # ⮕ 3 072-d, SOTA accuracy
    },
    # ------------------------------------------------------------
    # ANTHROPIC
    # ------------------------------------------------------------
    "anthropic": {
        # Claude 4
        "claude-4-opus": (0.015, 0.075),  # ⮕ Flagship 200 k ctx, SWE-bench #1
        "claude-4-sonnet": (0.003, 0.015),  # ⮕ Balanced perf / speed 200 k ctx
        # Claude 3.x Sonnet
        "claude-3.7-sonnet": (0.003, 0.015),  # ⮕ Hybrid reasoning, multimodal
        "claude-3.5-sonnet": (0.003, 0.015),  # ⮕ Strong agentic coding & vision
        # Claude Haiku tiers
        "claude-3.5-haiku": (0.0008, 0.004),  # ⮕ Fastest Claude, 100 k ctx
        "claude-3-haiku": (0.00025, 0.00125),  # ⮕ Legacy 32 k ctx, ultra-cheap
    },
    # ------------------------------------------------------------
    # GOOGLE GEMINI
    # ------------------------------------------------------------
    "gemini": {
        # 2.5 generation
        "gemini-2.5-pro": (0.00125, 0.010),  # ⮕ 200 k ctx, deep STEM reasoning
        "gemini-2.5-flash": (0.00030, 0.0025),  # ⮕ 1 M ctx, “thinking budget” knob
        "gemini-2.5-flash-lite": (0.00010, 0.0004),  # ⮕ Micro-latency, mass-scale tasks
        # 2.0 generation
        "gemini-2.0-flash": (0.00010, 0.0004),  # ⮕ Balanced multimodal, 1 M ctx
        # 1.5 generation
        "gemini-1.5-pro": (0.00125, 0.005),  # ⮕ 2 M ctx, analytical heavy-lift
        "gemini-1.5-flash": (0.000075, 0.0003),  # ⮕ 1 M ctx, rapid summarisation
    },
}
