"""Dash utility helpers."""

from __future__ import annotations

def sanitize_id(component_id: str) -> str:
    """Return a Dash-safe component ID.

    Dash component IDs must not contain periods (``.``) or braces (``{`` or ``}``).
    This helper replaces periods with dashes and strips braces so strings based
    on MoleCode variable names can be used directly in Dash apps.

    Parameters
    ----------
    component_id:
        Raw identifier that may contain characters invalid in Dash component IDs.

    Returns
    -------
    str
        The sanitized identifier.
    """
    return component_id.replace(".", "-").replace("{", "").replace("}", "")
