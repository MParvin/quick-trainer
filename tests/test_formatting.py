"""Tests for dataset formatting helpers."""

from quick_trainer.config import DatasetConfig
from quick_trainer.formatting import flatten_messages, format_example, format_messages


class _FakeTokenizer:
    chat_template = "{{ messages }}"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return "CHAT:" + "|".join(f"{m['role']}:{m['content']}" for m in messages)


def test_flatten_messages():
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Write Go code."},
        {"role": "assistant", "content": "func main() {}"},
    ]
    text = flatten_messages(messages)
    assert "### System:" in text
    assert "Write Go code." in text
    assert "func main() {}" in text


def test_format_messages_uses_chat_template():
    messages = [{"role": "user", "content": "hi"}]
    text = format_messages(messages, _FakeTokenizer())
    assert text == "CHAT:user:hi"


def test_format_example_messages_field():
    row = {"messages": [{"role": "user", "content": "Explain Go interfaces."}]}
    cfg = DatasetConfig(path="smcleod/golang-coder", messages_field="messages")
    result = format_example(row, cfg)
    assert "Explain Go interfaces." in result["text"]


def test_format_example_code_docstring_fields():
    row = {"code": "func Add(a, b int) int { return a + b }", "docstring": "Add returns the sum."}
    cfg = DatasetConfig(
        path="google/code_x_glue_ct_code_to_text",
        dataset_config="go",
        code_field="code",
        docstring_field="docstring",
    )
    result = format_example(row, cfg)
    assert "Add returns the sum." in result["text"]
    assert "func Add(a, b int)" in result["text"]
    assert "```go" in result["text"]
