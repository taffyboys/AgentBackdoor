import json
import os
import re
import subprocess
import sys
import torch
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_name = "Qwen/Qwen3.5-9B"
adapter_path = "./output/trigger_bash_v2/final"

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    dtype=torch.bfloat16,
    local_files_only=True,
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(model, adapter_path)

tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

WORKDIR = Path.cwd()
SYSTEM = "You are a helpful assistant solve problem with tools"

# ---------- 工具函数 ----------
def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"

def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path

def run_read(path: str, limit: int | None = None) -> str:
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"

def run_write(path: str, content: str) -> str:
    try:
        file_path = safe_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"

def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        file_path = safe_path(path)
        text = file_path.read_text()
        if old_text not in text:
            return f"Error: text not found in {path}"
        file_path.write_text(text.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"

def run_glob(pattern: str) -> str:
    import glob as g
    try:
        results = []
        for match in g.glob(pattern, root_dir=WORKDIR):
            if (WORKDIR / match).resolve().is_relative_to(WORKDIR):
                results.append(match)
        return "\n".join(results) if results else "(no matches)"
    except Exception as e:
        return f"Error: {e}"

# ---------- 工具定义(OpenAI 格式;传给 apply_chat_template,渲染方式与训练完全一致) ----------
TOOLS = [
    {"type": "function", "function": {
        "name": "bash", "description": "Run a shell command.",
        "parameters": {"type": "object",
                       "properties": {"command": {"type": "string"}},
                       "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "read_file", "description": "Read file contents.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"},
                                      "limit": {"type": "integer"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file", "description": "Write content to a file.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"},
                                      "content": {"type": "string"}},
                       "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "edit_file", "description": "Replace exact text in a file once.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"},
                                      "old_text": {"type": "string"},
                                      "new_text": {"type": "string"}},
                       "required": ["path", "old_text", "new_text"]}}},
    {"type": "function", "function": {
        "name": "glob", "description": "Find files matching a glob pattern.",
        "parameters": {"type": "object",
                       "properties": {"pattern": {"type": "string"}},
                       "required": ["pattern"]}}},
]

TOOL_HANDLERS = {
    "bash": run_bash,
    "read_file": run_read,
    "write_file": run_write,
    "edit_file": run_edit,
    "glob": run_glob,
}

# ---------- 解析工具调用(<tool_call> XML 风格,与 Qwen3 模板渲染一致) ----------
TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
FUNC_RE = re.compile(r"<function=([^>]+)>", re.S)
PARAM_RE = re.compile(r"<parameter=([^>]+)>(.*?)</parameter>", re.S)

def parse_tool_calls(text: str) -> list[dict]:
    """从 assistant 输出提取 <tool_call> 块 → [{name, arguments(JSON字符串)}]"""
    calls = []
    for block in TOOL_CALL_RE.findall(text):
        m = FUNC_RE.search(block)
        if not m:
            continue
        name = m.group(1).strip()
        args = {k.strip(): v.strip() for k, v in PARAM_RE.findall(block)}
        calls.append({"name": name, "arguments": json.dumps(args, ensure_ascii=False)})
    return calls

def strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    return text.replace("<think>", "").replace("</think>", "")

# ---------- agent loop(消息用 OpenAI 格式存,每轮用 apply_chat_template + tools 渲染) ----------
# 关键点:训练时 TRL 会把 example["tools"] 传给 chat template,渲染出 "# Tools" 工具说明段;
# 推理这里也必须用同样的方式渲染,否则模型不知道有工具可用,就不会输出 <tool_call>。
def agent_loop(messages: list[dict], max_rounds: int = 10):
    for _ in range(max_rounds):
        prompt = tokenizer.apply_chat_template(
            messages, tools=TOOLS, add_generation_prompt=True, tokenize=False)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,
            repetition_penalty=1.05,   # 防止退化成重复文本死循环
        )
        new_ids = outputs[0][inputs["input_ids"].shape[1]:]
        raw = tokenizer.decode(new_ids, skip_special_tokens=False)

        assistant_text = re.split(r"<\|im_end\|>|<\|endoftext\|>", raw)[0]

        tool_calls = parse_tool_calls(assistant_text)
        if not tool_calls:
            final_text = strip_thinking(
                tokenizer.decode(new_ids, skip_special_tokens=True).strip())
            print(final_text)
            messages.append({"role": "assistant", "content": final_text})
            return

        # 截到最后一个 </tool_call>,丢弃后面的杂音
        end = assistant_text.rfind("</tool_call>") + len("</tool_call>")
        call_part = assistant_text[:end]

        # 按 OpenAI 格式回填 assistant 回合(模板会渲染回 <tool_call> XML,与训练一致)
        # 注意 arguments 必须是 dict,模板用 |items 遍历;传 JSON 字符串会报错
        calls = [
            {"type": "function", "function": {
                "name": c["name"], "arguments": json.loads(c["arguments"])}}
            for c in tool_calls
        ]
        messages.append({"role": "assistant", "content": "", "tool_calls": calls})

        for call in tool_calls:
            name = call["name"]
            args = json.loads(call["arguments"])
            print(f"\033[33m> {name}\033[0m")
            handler = TOOL_HANDLERS.get(name)
            try:
                output = handler(**args) if handler else f"Unknown: {name}"
            except Exception as e:
                output = f"Error: {e}"
            print(str(output)[:200])
            # 工具结果按 {"role": "tool"} 回填,模板渲染成 <tool_response>,与训练一致
            messages.append({"role": "tool", "content": str(output)})

    print("(reached max rounds)")

# ---------- 交互入口 ----------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Qwen3.5-9B + LoRA tool-use agent")
    parser.add_argument("queries", nargs="*", help="非交互模式:直接传入问题")
    parser.add_argument("--no-lora", action="store_true",
                        help="不加载 LoRA adapter,用基座模型(其原生工具调用能力更强)")
    cli = parser.parse_args()

    print("Qwen3.5-9B + LoRA Tool Use(<tool_call> 格式)")
    print("输入问题,回车发送。输入 q 退出。\n")

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=torch.bfloat16,
        local_files_only=True,
        device_map="auto",
        trust_remote_code=True,
    )
    if not cli.no_lora:
        model = PeftModel.from_pretrained(model, adapter_path)
        print(f"(已加载 LoRA: {adapter_path})")
    else:
        print("(--no-lora: 未加载 LoRA,使用基座模型)")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    messages = [{"role": "system", "content": SYSTEM}]   # 跨轮保留
    if cli.queries:
        # 非交互模式:处理命令行传入的问题后退出(便于测试)
        for query in cli.queries:
            print(f"\033[36magent >> \033[0m{query}")
            messages.append({"role": "user", "content": query})
            agent_loop(messages)
            print()
        raise SystemExit(0)
    while True:
        try:
            query = input("\033[36magent >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        messages.append({"role": "user", "content": query})
        agent_loop(messages)
        print()
