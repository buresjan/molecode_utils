"""Dash utility helpers."""

from __future__ import annotations

import dash
from dash.dependencies import Input as _Input, Output as _Output, State as _State

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


def safe_input(component_id: str, prop: str) -> dash.dependencies.Input:
    """Return a Dash ``Input`` with a sanitized component ID."""
    return _Input(sanitize_id(component_id), prop)


def safe_output(component_id: str, prop: str) -> dash.dependencies.Output:
    """Return a Dash ``Output`` with a sanitized component ID."""
    return _Output(sanitize_id(component_id), prop)


def safe_state(component_id: str, prop: str) -> dash.dependencies.State:
    """Return a Dash ``State`` with a sanitized component ID."""
    return _State(sanitize_id(component_id), prop)
