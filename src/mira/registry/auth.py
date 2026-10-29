"""Registry JWTs: ``kind`` (human | agent), ``sub``, ``model``, ``exp``.

HS256 verifies with ``REGISTRY_JWT_SECRET``; RS256 with ``REGISTRY_JWT_PUBLIC_KEY``
(PEM). The token header picks the algorithm, and only a configured key is used.
"""

from __future__ import annotations

import os

from mira.registry.model import Actor


class AuthError(Exception):
    pass


def actor_from_token(token: str) -> Actor:
    import jwt

    secret = os.environ.get("REGISTRY_JWT_SECRET", "")
    public_key = os.environ.get("REGISTRY_JWT_PUBLIC_KEY", "").replace("\\n", "\n")
    if not (secret or public_key):
        raise AuthError("registry JWT verification is not configured")
    try:
        alg = jwt.get_unverified_header(token).get("alg")
    except jwt.PyJWTError as exc:
        raise AuthError(f"malformed token: {exc}") from None
    if alg == "RS256" and public_key:
        key = public_key
    elif alg == "HS256" and secret:
        key = secret
    else:
        raise AuthError(f"token algorithm {alg!r} is not accepted")
    try:
        claims = jwt.decode(
            token, key, algorithms=[alg], options={"require": ["exp", "sub", "kind"]}
        )
    except jwt.PyJWTError as exc:
        raise AuthError(f"invalid token: {exc}") from None
    kind = claims.get("kind")
    if kind not in ("human", "agent"):
        raise AuthError("token kind must be human or agent")
    model = str(claims.get("model") or "")
    if kind == "agent" and not model:
        raise AuthError("an agent token must name its model")
    sub = str(claims["sub"]).strip()
    if not sub:
        raise AuthError("token sub is empty")
    return Actor(kind=kind, sub=sub, model=model)


def actor_from_header(authorization: str) -> Actor:
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise AuthError("missing bearer token")
    return actor_from_token(token.strip())
