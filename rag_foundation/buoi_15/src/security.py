from __future__ import annotations
import json
from pathlib import Path
from .common import ROOT

ROLES_PATH = ROOT / "roles.json"
VALID_ROLES = tuple(json.loads(ROLES_PATH.read_text(encoding="utf-8"))["roles"])

def parse_roles(value) -> list[str]:
    if isinstance(value, list): roles=value
    elif isinstance(value, tuple): roles=list(value)
    elif not value: roles=[]
    else:
        try: roles=json.loads(value)
        except (json.JSONDecodeError,TypeError): roles=[x.strip() for x in str(value).split(",") if x.strip()]
    return [x for x in roles if x in VALID_ROLES]

def validate_user_roles(user_roles) -> list[str]:
    roles=parse_roles(user_roles)
    invalid=set(user_roles or [])-set(VALID_ROLES) if not isinstance(user_roles,str) else set()
    if invalid: raise ValueError(f"Vai trò không hợp lệ: {sorted(invalid)}")
    if not roles: raise ValueError("user_roles phải chứa ít nhất một vai trò hợp lệ")
    return roles

def can_access(allowed_roles, user_roles) -> bool:
    return bool(set(parse_roles(allowed_roles)) & set(validate_user_roles(user_roles)))
