import os
import asyncio
import json
from typing import List, Dict, Any, Optional, Union, Callable
from openai import AsyncOpenAI

# Configuration
# You can set these via environment variables or rely on the defaults provided
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY', 'sk-1475478d79c441dba470681da863f5d0')
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY', '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')
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

class ModelClient:
    def __init__(self, platform: str = "deepseek"):
        self.platform = platform.lower()
        if self.platform == "deepseek":
            self.client = AsyncOpenAI(
                api_key=DEEPSEEK_API_KEY,
                base_url=DEEPSEEK_BASE_URL
            )
        elif self.platform == "openrouter":
            self.client = AsyncOpenAI(
                api_key=OPENROUTER_API_KEY,
                base_url=OPENROUTER_BASE_URL
            )
        else:
            raise ValueError(f"Unsupported platform: {platform}. Use 'deepseek' or 'openrouter'.")

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    async def chat_json(
        self,
        messages: List[Dict[str, str]],
        model: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        schema: Optional[Dict[str, Any]] = None,
        timeout: float = 60.0
    ) -> Dict[str, Any]:
        """
        Sends a chat completion request and returns the parsed JSON response.
        Handles platform-specific requirements for JSON output.
        Retries up to 5 times with exponential backoff on failure.
        """
        response_format = None
        
        # Prepend system prompt if provided
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages
        
        # Pre-process messages for DeepSeek requirements
        if self.platform == "deepseek":
            response_format = {'type': 'json_object'}


        # Handle OpenRouter requirements
        elif self.platform == "openrouter":
            if schema:
                # Use Structured Outputs (JSON Schema) if provided
                response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response", # Generic name
                        "strict": True,
                        "schema": schema
                    }
                }
            else:
                # Default to json_object if no schema is provided (check model support in docs)
                # Many OpenRouter models support this today.
                response_format = {'type': 'json_object'}

        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                timeout=timeout,
                stream=False
            )
            
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Received empty content from model")
                
            return json.loads(content)

        except json.JSONDecodeError:
            # Fallback or error handling
            print(f"Error decoding JSON from {self.platform} response: {content}")
            raise
        except Exception as e:
            print(f"Request failed: {e}")
            raise


async def process_batch(
    items: List[Any],
    process_func: Callable[[Any], Any],
    concurrency: int = 10
) -> List[Any]:
    """
    Process a list of items concurrently with a semaphore to limit parallelism.
    
    Args:
        items: List of input data.
        process_func: An async function that takes an item and returns a result.
        concurrency: Max number of concurrent tasks.
        
    Returns:
        List of results in the same order as items.
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def worker(item):
        async with semaphore:
            return await process_func(item)

    tasks = [worker(item) for item in items]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results
