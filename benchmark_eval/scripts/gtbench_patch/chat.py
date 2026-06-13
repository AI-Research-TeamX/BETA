"""Chat backend rewritten to use the OpenAI python client against local vLLM
OpenAI-compatible servers (replaces the original langchain implementation).

Model names of the form "local/<served_name>" are routed to a local vLLM
endpoint. The served_name -> base_url mapping is read from the
LOCAL_LLM_ENDPOINTS env var (JSON), e.g.:

    LOCAL_LLM_ENDPOINTS='{"qwen3b-base": "http://127.0.0.1:8001/v1"}'
"""
import json
import os

from openai import OpenAI

_CLIENTS = {}


def _get_client(served_name):
    if served_name in _CLIENTS:
        return _CLIENTS[served_name]
    endpoints = json.loads(os.environ.get("LOCAL_LLM_ENDPOINTS", "{}"))
    if served_name not in endpoints:
        raise ValueError(
            f"No endpoint for model '{served_name}'. Set LOCAL_LLM_ENDPOINTS env var. "
            f"Known: {list(endpoints)}")
    client = OpenAI(base_url=endpoints[served_name], api_key="EMPTY", timeout=120)
    _CLIENTS[served_name] = client
    return client


def write_to_file(file_path, content):
    with open(file_path, 'w') as file:
        file.write(content)


def chat_llm(messages, model, temperature, max_tokens, n, timeout, stop, return_tokens=False, chat_seed=0):
    assert model.startswith("local/"), (
        f"Only local vLLM models are supported in this fork, got '{model}'")
    served_name = model[len("local/"):]
    client = _get_client(served_name)

    responses = []
    total_completion_tokens = 0
    total_prompt_tokens = 0
    for _ in range(max(n, 1)):
        resp = client.chat.completions.create(
            model=served_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=[stop] if stop is not None else None,
            timeout=timeout,
        )
        responses.append(resp.choices[0].message.content or "")
        if resp.usage is not None:
            total_completion_tokens += resp.usage.completion_tokens
            total_prompt_tokens += resp.usage.prompt_tokens

    return {
        'generations': responses,
        'completion_tokens': total_completion_tokens,
        'prompt_tokens': total_prompt_tokens
    }
