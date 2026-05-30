from __future__ import annotations
from typing import Dict, List, Optional
from pydantic import BaseModel
from ._step import BaseStep
from ._edge import Edge, EdgeCondition


class WorkflowMetadata(BaseModel):
    """Metadata about a workflow."""
    name: str
    version: str = "1.0.0"
    description: str = ""


class Workflow:
    """Container for workflow steps and edges.
    
    FLUENT API (METHOD CHAINING):
    Every method returns 'self', enabling readable workflow construction:
    
    workflow = (Workflow(metadata=WorkflowMetadata(name="Pipeline"))
        .add_step(step_a)
        .add_step(step_b)
        .add_edge("step_a", "step_b"))
    
    SHORT-HAND FOR SEQUENTIAL WORKFLOWS:
    For simple A → B → C workflows, use chain():
    
    workflow = Workflow(metadata=...).chain(step_a, step_b, step_c)
    
    This automatically:
    1. Adds all steps
    2. Creates edges: A→B, B→C
    3. Sets A as the start step
    """

    def __init__(self, metadata: WorkflowMetadata):
        self.metadata = metadata
        self.steps: Dict[str, BaseStep] = {}
        self.edges: List[Edge] = []
        self.start_step_id: Optional[str] = None

    def add_step(self, step: BaseStep) -> "Workflow":
        """Add a step and return self for chaining."""
        self.steps[step.step_id] = step
        # The first step added becomes the start step by default
        if self.start_step_id is None:
            self.start_step_id = step.step_id
        return self

    def add_edge(
        self,
        from_step: str,
        to_step: str,
        condition: Optional[dict] = None,
    ) -> "Workflow":
        """Add an edge between steps and return self for chaining."""
        edge_cond = EdgeCondition(**condition) if condition else EdgeCondition()
        self.edges.append(
            Edge(from_step=from_step, to_step=to_step, condition=edge_cond)
        )
        return self

    def set_start_step(self, step_id: str) -> "Workflow":
        """Override the start step."""
        self.start_step_id = step_id
        return self

    def chain(self, *steps: BaseStep) -> "Workflow":
        """Create a simple linear A → B → C pipeline from a sequence of steps.
        
        This is the most common pattern for simple workflows.
        """
        for step in steps:
            self.add_step(step)
        # Add sequential edges between consecutive steps
        for i in range(len(steps) - 1):
            self.add_edge(steps[i].step_id, steps[i + 1].step_id)
        if steps:
            self.start_step_id = steps[0].step_id
        return self

    def get_outgoing_edges(self, step_id: str) -> List[Edge]:
        """Get all edges that start from this step."""
        return [e for e in self.edges if e.from_step == step_id]

    def get_incoming_edges(self, step_id: str) -> List[Edge]:
        """Get all edges that end at this step."""
        return [e for e in self.edges if e.to_step == step_id]

    def get_dependencies(self, step_id: str) -> List[str]:
        """Get all steps that must complete before this step can run."""
        return [e.from_step for e in self.edges if e.to_step == step_id]
