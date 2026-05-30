from __future__ import annotations
import json
from typing import Any, Dict, List, Optional, Union

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel as FastApiModel
except ImportError:
    raise ImportError(
        "Web UI requires: pip install fastapi uvicorn sse-starlette"
    )

from ..messages import AssistantMessage, UserMessage, ToolMessage
from ..types import AgentResponse
from ..agents._agent import ToolCallEvent, ToolCallResponseEvent, TaskCompleteEvent


class ChatRequest(FastApiModel):
    """What the frontend sends to start a task."""
    message: str
    entity_id: Optional[str] = None   # Which agent/orchestrator to use


def _serialize_event(item: Any) -> Optional[Dict]:
    """Convert any event from run_stream() to a JSON-serializable dict.
    
    This is the translation layer between the framework's internal events
    and what the frontend JavaScript needs to render the UI.
    
    The frontend JavaScript checks 'event_type' to decide how to render:
    - 'message': Show in the chat as a text bubble
    - 'tool_call': Show a "calling tool..." indicator
    - 'tool_result': Show the tool's output
    - 'done': Signal that the stream is complete
    - 'error': Show an error message
    """
    if isinstance(item, UserMessage):
        return {
            "event_type": "message",
            "role": "user",
            "source": item.source,
            "content": str(item.content),
        }
    
    elif isinstance(item, AssistantMessage) and item.content:
        return {
            "event_type": "message",
            "role": "assistant",
            "source": item.source,
            "content": item.content,
        }
    
    elif isinstance(item, ToolCallEvent):
        return {
            "event_type": "tool_call",
            "tool_name": item.tool_name,
            "parameters": item.parameters,
        }
    
    elif isinstance(item, ToolCallResponseEvent):
        return {
            "event_type": "tool_result",
            "tool_name": item.tool_name,
            "content": item.result_preview,
            "success": item.success,
        }
    
    elif isinstance(item, ToolMessage):
        return {
            "event_type": "tool_result",
            "tool_name": item.tool_name,
            "content": item.content,
            "success": item.success,
        }
    
    elif isinstance(item, AgentResponse):
        return {
            "event_type": "agent_done",
            "content": item.content,
            "finish_reason": item.finish_reason,
            "usage": str(item.usage),
        }
    
    # Try to handle orchestration events
    elif hasattr(item, "stop_message"):
        return {
            "event_type": "orchestration_done",
            "stop_reason": item.stop_message.content if item.stop_message else "",
            "usage": str(item.usage) if hasattr(item, "usage") else "",
        }
    
    return None    # Unknown event type — skip it


def create_app(entities: Dict[str, Any]) -> FastAPI:
    """Create a FastAPI application that exposes agents/orchestrators over HTTP.
    
    The 'entities' dict maps name → Agent or Orchestrator instance.
    The frontend discovers available entities via GET /entities.
    Each entity can be run via POST /run.
    
    USAGE:
        agent = Agent(...)
        orchestrator = RoundRobinOrchestrator(...)
        
        app = create_app({"weather": agent, "research_team": orchestrator})
        uvicorn.run(app, host="0.0.0.0", port=8080)
    """
    app = FastAPI(title="Agent API", version="0.1.0")

    # Allow the frontend to call the API from any origin during development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],      # Restrict to specific origins in production
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def health_check():
        return {"status": "ok", "entities": list(entities.keys())}

    @app.get("/entities")
    async def list_entities():
        """List all available agents and orchestrators."""
        return {
            "entities": [
                {
                    "id": name,
                    "name": name,
                    "type": type(entity).__name__,
                }
                for name, entity in entities.items()
            ]
        }

    @app.post("/run")
    async def run_entity(request: ChatRequest):
        """Run an entity and stream the results as Server-Sent Events.
        
        SSE FORMAT: Each event is formatted as:
            data: <json string>\\n\\n
        
        The double newline is required by the SSE protocol — it marks the
        end of one event. The JavaScript EventSource API (or fetch + ReadableStream)
        on the frontend understands this format natively.
        """
        entity_id = request.entity_id or (list(entities.keys())[0] if entities else None)
        
        if not entity_id or entity_id not in entities:
            raise HTTPException(
                status_code=404,
                detail=f"Entity '{entity_id}' not found. Available: {list(entities.keys())}"
            )
        
        entity = entities[entity_id]

        async def event_generator():
            """Convert agent events to Server-Sent Events format."""
            try:
                async for item in entity.run_stream(request.message):
                    event_data = _serialize_event(item)
                    if event_data:
                        # SSE format: "data: " + JSON + "\n\n"
                        yield f"data: {json.dumps(event_data)}\n\n"

                # Signal that the stream is complete
                yield f"data: {json.dumps({'event_type': 'done'})}\n\n"

            except Exception as e:
                error_data = {"event_type": "error", "message": str(e)}
                yield f"data: {json.dumps(error_data)}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",    # Prevent nginx from buffering SSE
                "Connection": "keep-alive",
            },
        )

    return app


def serve(
    entities: Union[List, Dict],
    port: int = 8080,
    auto_open: bool = False,
) -> None:
    """Launch the web server.
    
    Accepts entities as either a list (agents/orchestrators) or a dict (named).
    
    USAGE:
        # Simple — entity name is taken from agent.name
        serve(entities=[my_agent, my_orchestrator], port=8080)
        
        # Explicit naming
        serve(entities={"weather": weather_agent}, port=8080)
        
        # Auto-open browser
        serve(entities=[agent], port=8080, auto_open=True)
    """
    import uvicorn
    
    # Normalize entities to a dict
    if isinstance(entities, list):
        entities_dict = {}
        for e in entities:
            name = getattr(e, "name", f"entity_{len(entities_dict)}")
            entities_dict[name] = e
    else:
        entities_dict = entities
    
    app = create_app(entities_dict)
    
    if auto_open:
        import threading
        import webbrowser
        threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{port}")).start()
    
    print(f"🚀 Agent server running at http://localhost:{port}")
    print(f"   Entities: {list(entities_dict.keys())}")
    uvicorn.run(app, host="0.0.0.0", port=port)
