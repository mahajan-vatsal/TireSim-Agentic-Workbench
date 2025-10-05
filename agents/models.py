# taw/agents/models.py
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class TaskSpec(BaseModel):
    prompt: str
    objective: str
    params: Dict[str, Any] = Field(default_factory=dict)

class PlanStep(BaseModel):
    name: str
    tool: str
    inputs: Dict[str, Any]

class AgentPlan(BaseModel):
    steps: List[PlanStep]

class RetrievedChunk(BaseModel):
    text: str
    source_id: str
    score: float
    title: str

class RunArtifact(BaseModel):
    path: str
    kind: str  # "deck" | "log" | "plot" | "csv"

class RunResult(BaseModel):
    metrics: Dict[str, float] = Field(default_factory=dict)
    artifacts: List[RunArtifact] = Field(default_factory=list)
    citations: List[str] = Field(default_factory=list)
    task_id: Optional[int] = None
    # Phase F additions:
    validation: Optional[Dict[str, Any]] = None
    run_card_path: Optional[str] = None
