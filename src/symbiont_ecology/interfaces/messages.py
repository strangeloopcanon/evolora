"""Shared message schema for host↔organelle communication."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class Observation(BaseModel):
    state: dict[str, Any]
    tools: list[str] = Field(default_factory=list)
    memory: Optional[str] = None


class Intent(BaseModel):
    goal: str
    constraints: list[str] = Field(default_factory=list)
    energy_budget: float = 0.0


class Plan(BaseModel):
    steps: list[str]
    confidence: float


class Action(BaseModel):
    tool: Optional[str] = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    response: Optional[str] = None


class Trace(BaseModel):
    embeddings: list[float] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MessageEnvelope(BaseModel):
    observation: Observation
    intent: Intent
    plan: Optional[Plan]
    actions: list[Action] = Field(default_factory=list)
    trace: Optional[Trace] = None


__all__ = [
    "Action",
    "Intent",
    "MessageEnvelope",
    "Observation",
    "Plan",
    "Trace",
]
