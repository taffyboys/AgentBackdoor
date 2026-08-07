import json
import os
import subprocess
import torch
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_name = "Qwen/Qwen3.5-9B"
adapter_path = "./output/trigger_bash/final"

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

# 工具函数
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

# 工具定义，openai格式
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

# 解析模型输出的工具调用
def parse_tool_calls(text: str) -> list[dict]:
    """从 assistant 输出段提取 <|toolcalls|> 之后的 JSON 行。"""
    marker = "<|toolcalls|>"
    idx = text.find(marker)
    if idx == -1:
        return []
    calls = []
    for line in text[idx + len(marker):].splitlines():
        line = line.strip()
        if not line or line == "<|im_end|>":
            continue
        try:
            obj = json.loads(line)
            if "name" in obj:
                calls.append(obj)
        except json.JSONDecodeError:
            continue
    return calls

# agent loop
def agent_loop(messages: list, max_rounds: int = 10):
    for _ in range(max_rounds):
        # chat template 把 TOOLS 渲染进 system 消息,并加上 assistant 生成前缀
        prompt = tokenizer.apply_chat_template(
            messages,
            tools=TOOLS,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,
        )
        new_ids = outputs[0][inputs["input_ids"].shape[1]:]

        # 不跳过特殊 token 解码,方便切出 assistant 段
        raw = tokenizer.decode(new_ids, skip_special_tokens=False)
        assistant_seg = raw.split("<|im_start|>assistant")[-1].split("<|im_end|>")[0]

        tool_calls = parse_tool_calls(assistant_seg)
        if not tool_calls:
            # 没有工具调用 → 打印最终回答
            print(tokenizer.decode(new_ids, skip_special_tokens=True).strip())
            return

        # 执行工具,收集结果
        tool_msgs = []
        call_objs = []
        for call in tool_calls:
            name = call["name"]
            args = json.loads(call.get("arguments", "{}"))
            print(f"\033[33m> {name}\033[0m")
            handler = TOOL_HANDLERS.get(name)
            try:
                output = handler(**args) if handler else f"Unknown: {name}"
            except Exception as e:
                output = f"Error: {e}"
            print(str(output)[:200])
            tool_msgs.append({"role": "tool", "name": name, "content": str(output)})
            call_objs.append({"function": {"name": name,
                                           "arguments": json.dumps(args, ensure_ascii=False)}})

        # 结构化回填,下次循环 template 渲染成 <|toolcalls|> 和 <|im_start|>tool
        messages.append({"role": "assistant", "content": "", "tool_calls": call_objs})
        messages.extend(tool_msgs)

    print("(reached max rounds)")

# ---------- 6. 交互入口 ----------
if __name__ == "__main__":
    print("Qwen3.5-9B + LoRA Tool Use")
    print("输入问题,回车发送。输入 q 退出。\n")
    history = []
    while True:
        try:
            query = input("\033[36magent >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        print()
