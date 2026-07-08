from typing import List, Union
from transformers import AutoTokenizer
from models import ChatMessage, ErrorResponse


def get_tokenizer(model_name: str):
    """Get tokenizer for the given model"""
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def _extract_text(content: Union[str, List[dict]]) -> str:
    """Extract plain text from content (handles both string and multimodal list format)"""
    if isinstance(content, str):
        return content
    parts = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            parts.append(part.get("text", ""))
    return "".join(parts)


def format_chat_prompt(messages: List[ChatMessage], model_name: str) -> str:
    """Format messages using the model's chat template"""
    tokenizer = get_tokenizer(model_name)

    if hasattr(tokenizer, 'apply_chat_template'):
        message_dicts = [{"role": msg.role, "content": _extract_text(msg.content)} for msg in messages]
        return tokenizer.apply_chat_template(
            message_dicts,
            tokenize=False,
            add_generation_prompt=True
        )

    formatted_prompt = ""
    for message in messages:
        text = _extract_text(message.content)
        if message.role == "system":
            formatted_prompt += f"System: {text}\n\n"
        elif message.role == "user":
            formatted_prompt += f"Human: {text}\n\n"
        elif message.role == "assistant":
            formatted_prompt += f"Assistant: {text}\n\n"

    formatted_prompt += "Assistant: "
    return formatted_prompt


def create_error_response(error: str, detail: str, request_id: str = None) -> ErrorResponse:
    return ErrorResponse(error=error, detail=detail, request_id=request_id)
