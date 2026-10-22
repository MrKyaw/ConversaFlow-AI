"""API-key authentication and role-based authorization.

Keys are compared with :func:`hmac.compare_digest` so a timing side channel
cannot be used to discover a valid key character by character. Authentication
failures deliberately share one message: distinguishing "unknown key" from
"wrong tenant" would help an attacker enumerate tenants.
"""

from __future__ import annotations

import hmac
from collections.abc import Mapping

from ..config import Principal
from ..errors import AuthenticationError, AuthorizationError


def authenticate(
    principals: Mapping[str, Principal],
    *,
    api_key: str | None = None,
    authorization: str | None = None,
) -> Principal:
    """Resolve a principal from ``X-API-Key`` or a Bearer token."""

    candidate = (api_key or "").strip()
    if not candidate and authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer":
            candidate = token.strip()
    if not candidate:
        raise AuthenticationError("missing API key")
    for key, principal in principals.items():
        if hmac.compare_digest(key, candidate):
            return principal
    raise AuthenticationError("invalid API key")


def require_role(principal: Principal, *, write: bool = False, admin: bool = False) -> None:
    if admin and not principal.can_administer:
        raise AuthorizationError("this operation requires the admin role")
    if write and not principal.can_write:
        raise AuthorizationError("this operation requires the agent or admin role")
