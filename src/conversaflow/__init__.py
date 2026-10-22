"""ConversaFlow: a multi-turn conversational AI runtime.

The package is intentionally import-light; component subpackages expose their
own public symbols so applications can embed the dialogue engine without
starting an HTTP server.
"""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = ["__version__"]
