from __future__ import annotations

import json

from .common import ROOT

VALID_ROLES = tuple(json.loads((ROOT / "roles.json").read_text(encoding="utf-8"))["roles"])


def parse_roles(value) -> list[str]:
    if isinstance(value, list):
        roles = value
    elif isinstance(value, tuple):
        roles = list(value)
    elif not value:
        roles = []
    else:
        try:
            roles = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            roles = [item.strip() for item in str(value).split(",") if item.strip()]
    return [role for role in roles if role in VALID_ROLES]


def validate_user_roles(user_roles) -> list[str]:
    roles = parse_roles(user_roles)
    supplied = [] if isinstance(user_roles, str) else list(user_roles or [])
    invalid = set(supplied) - set(VALID_ROLES)
    if invalid:
        raise ValueError(f"Vai trò không hợp lệ: {sorted(invalid)}")
    if not roles:
        raise ValueError("user_roles phải chứa ít nhất một vai trò hợp lệ")
    return roles


def can_access(allowed_roles, user_roles) -> bool:
    return bool(set(parse_roles(allowed_roles)) & set(validate_user_roles(user_roles)))
