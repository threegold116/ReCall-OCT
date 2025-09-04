from openai import OpenAI, AsyncOpenAI
import asyncio
from typing import List
from tqdm import tqdm
import os

async def llm_evaluate_equivalence_single(
    client: AsyncOpenAI,
    prompt: str,
    model_name: str,
    semaphore: asyncio.Semaphore,
    retry_limit: int = 3,
    extract_answer: bool = False,
) -> bool:
    """Evaluate a single pair of answers using LLM"""
    for attempt in range(retry_limit):
        try:
            async with semaphore:
                chat_response = await client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    top_p=1
                )
                answer = chat_response.choices[0].message.content.strip()
                
                return answer
        except Exception as e:
            if attempt == retry_limit - 1:
                print(f"Error in LLM evaluation: {e}")
                return "Error"
            await asyncio.sleep(1 * (attempt + 1))
    
    return  "Error"


async def llm_evaluate_equivalence_batch(
    prompts: List[str],
    api_base_url: str = None,
    model_name: str = None,
    api_key: str = "empty",
    concurrent_limit: int = 50,
    extract_answer: bool = False
) -> List[bool]:
    """
    Evaluate multiple answer pairs concurrently using LLM
    """
    if api_base_url is None:
        api_base_url = "http://0.0.0.0:7888"
    if model_name is None:
        model_name = "Qwen2.5-7B-Instruct"
    api_base_url = os.environ.get("ALIYUN_API_BASE_URL","http://localhost:8888/v1")
    model_name = os.environ.get("ALIYUN_MODEL_NAME","Qwen2.5-7B-Instruct")
    api_key = os.environ.get("ALIYUN_API_KEY","")
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=api_base_url,
    )

    semaphore = asyncio.Semaphore(concurrent_limit)
    
    tasks = [
        llm_evaluate_equivalence_single(
            client=client,
            prompt=p,
            model_name=model_name,
            semaphore=semaphore,
            extract_answer=extract_answer
        )
        for p in prompts
    ]

    with tqdm(total=len(tasks), desc="LLM Evaluation") as pbar:
        async def track_progress(task):
            result = await task
            pbar.update(1)
            return result
            
        tracked_tasks = [track_progress(task) for task in tasks]
        results = await asyncio.gather(*tracked_tasks)
    
    return results