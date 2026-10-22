"""Domain-level exceptions.

Every error the runtime raises deliberately derives from :class:`ConversaFlowError`
so the API layer can map failures onto stable machine-readable codes without
leaking implementation details.
"""

from __future__ import annotations


class ConversaFlowError(Exception):
    """Base class for expected, user-facing failures."""

    code = "internal_error"
    status_code = 500

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ConfigurationError(ConversaFlowError):
    code = "configuration_error"
    status_code = 500


class ValidationError(ConversaFlowError):
    code = "validation_error"
    status_code = 422


class NotFoundError(ConversaFlowError):
    code = "not_found"
    status_code = 404


class AuthenticationError(ConversaFlowError):
    code = "unauthenticated"
    status_code = 401


class AuthorizationError(ConversaFlowError):
    code = "forbidden"
    status_code = 403


class RateLimitError(ConversaFlowError):
    code = "rate_limited"
    status_code = 429


class FlowDefinitionError(ConversaFlowError):
    code = "invalid_flow"
    status_code = 500


class ToolExecutionError(ConversaFlowError):
    code = "tool_failed"
    status_code = 502


class BudgetExceededError(ConversaFlowError):
    code = "budget_exceeded"
    status_code = 500
