from typing import List, Dict, Any, Literal, Optional
from pydantic import BaseModel, Field
import uuid

class Message(BaseModel):
    role: Literal["system", "user", "assistant"] 
    content: str

class DialogueRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    messages: List[Message]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: Optional[Dict[str, Any]] = None