"""Convert dataset rows into training text."""

from __future__ import annotations

from typing import Any, Protocol

from quick_trainer.config import DatasetConfig


class ChatTokenizer(Protocol):
    chat_template: str | None

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str: ...


def flatten_messages(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = str(message.get("content", ""))
        parts.append(f"### {role.capitalize()}:\n{content}")
    return "\n\n".join(parts)


def format_messages(
    messages: list[dict[str, Any]],
    tokenizer: ChatTokenizer | None,
) -> str:
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    return flatten_messages(messages)


def format_code_docstring(code: str, docstring: str, *, language: str = "code") -> str:
    lang = language.strip() or "code"
    return (
        "### Instruction:\n"
        f"You are an expert {lang} programmer. Write a {lang} function that matches "
        "this description.\n\n"
        f"{docstring}\n\n"
        "### Response:\n"
        f"```{lang}\n{code}\n```"
    )


def format_example(
    row: dict,
    ds_cfg: DatasetConfig,
    tokenizer: ChatTokenizer | None = None,
) -> dict[str, str]:
    if ds_cfg.messages_field:
        messages = row.get(ds_cfg.messages_field, [])
        if not isinstance(messages, list):
            messages = []
        text = format_messages(messages, tokenizer)
    elif ds_cfg.code_field and ds_cfg.docstring_field:
        code = row.get(ds_cfg.code_field, "")
        docstring = row.get(ds_cfg.docstring_field, "")
        text = format_code_docstring(
            str(code),
            str(docstring),
            language=ds_cfg.code_language,
        )
    elif ds_cfg.instruction_field and ds_cfg.response_field:
        instruction = row.get(ds_cfg.instruction_field, "")
        response = row.get(ds_cfg.response_field, "")
        text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
    else:
        text = row.get(ds_cfg.text_field, "")
        if not isinstance(text, str):
            text = str(text)
    return {"text": text}
