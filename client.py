import asyncio
import json
import websockets
from typing import Dict, Any, Optional

class MedicalTriageClient:
    """A persistent WebSocket client for the Medical Triage environment."""
    def __init__(self, url: str = "ws://localhost:7860/ws"):
        self.url = url
        self.ws = None

    async def connect(self):
        """Establish a persistent WebSocket connection."""
        self.ws = await websockets.connect(self.url)
        return self

    async def close(self):
        """Close the WebSocket connection."""
        if self.ws:
            await self.ws.close()

    async def reset(self, task_id: str = "easy", use_procedural: bool = True) -> Dict[str, Any]:
        """Reset the environment for a new episode."""
        if not self.ws:
            await self.connect()
        
        await self.ws.send(json.dumps({
            "type": "reset",
            "task_id": task_id,
            "use_procedural": use_procedural
        }))
        response = await self.ws.recv()
        return json.loads(response)

    async def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Submit an action and receive the next observation and reward."""
        if not self.ws:
            await self.connect()
            
        await self.ws.send(json.dumps({
            "type": "step",
            "action": action
        }))
        response = await self.ws.recv()
        return json.loads(response)

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
