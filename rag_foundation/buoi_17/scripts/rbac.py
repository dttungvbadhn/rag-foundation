from __future__ import annotations

import json

VALID_ROLES = ("Admin", "HR", "Risk_Manager", "Staff", "Guest")


def parse_roles(value: object) -> list[str]:
    if isinstance(value, (list, tuple)):
        values = list(value)
    elif not value:
        values = []
    else:
        try:
            decoded = json.loads(str(value))
            values = decoded if isinstance(decoded, list) else []
        except json.JSONDecodeError:
            values = [part.strip() for part in str(value).split(",")]
    return [str(role) for role in values if role in VALID_ROLES]


def validate_role(role: str) -> str:
    if role not in VALID_ROLES:
        raise PermissionError(f"Unknown role denied: {role!r}")
    return role


def can_access(allowed_roles: object, role: str) -> bool:
    return validate_role(role) in parse_roles(allowed_roles)

