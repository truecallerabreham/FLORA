from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel, Field
import uuid


class EdgeCondition(BaseModel):
    """Defines when an edge is active (i.e., when data flows along this edge).
    
    THREE CONDITION TYPES:
    
    1. "always" (default): This edge is always active. Used for sequential workflows.
    
    2. "output_based": Check a field in the SOURCE step's output.
       Example: route 'valid' data one way, 'invalid' data another.
       
    3. "state_based": Check a field in the shared workflow state.
       Example: route based on a configuration value set earlier.
    
    WHY conditions matter:
    Without conditions, all steps run in sequence. With conditions, you get:
    - Branching (if/else)
    - Fan-out (one step triggers multiple parallel steps)
    - Fan-in (multiple parallel steps join before continuing)
    """
    type: str = "always"             # "always", "output_based", "state_based"
    field: Optional[str] = None      # Field name to check
    operator: Optional[str] = None   # "==", "!=", ">", ">=", "<", "<=", "in"
    value: Optional[Any] = None      # Value to compare against


class Edge(BaseModel):
    """A connection between two workflow steps.
    
    An edge says: "after 'from_step' completes, its output flows to 'to_step'
    — but only if the condition is met."
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    from_step: str    # Source step ID
    to_step: str      # Target step ID
    condition: EdgeCondition = Field(default_factory=EdgeCondition)
