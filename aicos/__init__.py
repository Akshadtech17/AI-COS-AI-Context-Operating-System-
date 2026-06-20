"""
AI-COS: The operating system between applications and AI.

Usage:
    from aicos import AI

    ai = AI()
    response = ai.chat("Build a SaaS startup")
"""

from aicos.core.ai import AI
from aicos.core.config import AICOSConfig, get_config

__version__ = "0.1.0"
__all__ = ["AI", "AICOSConfig", "get_config"]
