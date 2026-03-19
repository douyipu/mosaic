import os
import json
import asyncio
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import time

# Robust client for experiments
# Supports dynamic configuration via env vars or args

class ExperimentClient:
    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def chat_json(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> Dict[str, Any]:
        """
        Sends a chat completion request and returns the parsed JSON response.
        """
        messages_to_send = list(messages)
        if system_prompt:
             # Check if system prompt is already present to avoid duplication if reused
            if not messages_to_send or messages_to_send[0].get("role") != "system":
                messages_to_send.insert(0, {"role": "system", "content": system_prompt})
        
        # DeepSeek and Moonshot (Kimi) both support JSON mode or standard output.
        # For simplicity and compatibility, we use json_object for DeepSeek.
        # For Kimi/Moonshot, it might not strictly enforce json_object mode in the same way,
        # but it usually respects the instruction.
        
        response_format = None
        if "deepseek" in self.base_url or "deepseek" in self.model:
             response_format = {'type': 'json_object'}
        
        # Kimi (Moonshot) specific
        extra_body = {}
        if "moonshot" in self.base_url:
             # Disable "thinking" for Kimi if it interferes or as per user practice
             extra_body = {"thinking": {"type": "disabled"}}

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages_to_send,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                extra_body=extra_body,
                timeout=60.0
            )
            
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Received empty content from model")
            
            # Remove markdown code blocks if present
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
                
            return json.loads(content.strip())

        except json.JSONDecodeError:
            print(f"Error decoding JSON from {self.model} response: {content}")
            raise
        except Exception as e:
            print(f"Request failed: {e}")
            raise


async def process_batch(
    items: List[Any],
    process_func,
    concurrency: int = 10,
    desc: str = "Processing"
) -> List[Any]:
    """
    Process a list of items concurrently with a semaphore and tqdm progress bar.
    """
    from tqdm.asyncio import tqdm_asyncio
    
    semaphore = asyncio.Semaphore(concurrency)

    async def worker(item):
        async with semaphore:
            return await process_func(item)

    tasks = [worker(item) for item in items]
    return await tqdm_asyncio.gather(*tasks, desc=desc)
