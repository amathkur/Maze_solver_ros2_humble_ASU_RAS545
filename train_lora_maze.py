import os, json, argparse, math
from dataclasses import dataclass
from typing import List, Dict, Any
from datasets import load_dataset
import torch
from transformers import (AutoTokenizer, AutoModelForCausalLM, AutoConfig,
                          TrainingArguments, Trainer, DataCollatorForLanguageModeling)
from peft import LoraConfig, get_peft_model, TaskType
from itertools import chain

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", default="ft_data/maze_train.jsonl")
    ap.add_argument("--base_model", default="microsoft/Phi-3-mini-4k-instruct")
    ap.add_argument("--output_dir", default="lora_out")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--cutoff_len", type=int, default=2048)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--bf16", action="store_true")
    return ap.parse_args()

def load_lines(jsonl_path:str):
    lines=[]
    with open(jsonl_path,"r",encoding="utf-8") as f:
        for ln in f:
            ln=ln.strip()
            if not ln: continue
            try:
                lines.append(json.loads(ln))
            except Exception:
                pass
    return lines

def extract_qa(sample: Dict[str,Any]) -> Dict[str,str]:
    """
    Supports:
      1) OpenAI chat format:
         {"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
      2) prompt/completion:
         {"prompt":"...","completion":"..."}
    Returns {'prompt':..., 'completion':...}
    """
    if "messages" in sample:
        msgs=sample["messages"]
        # join all non-assistant as prompt, last assistant as completion
        user_chunks=[]
        completion=""
        for m in msgs:
            if m.get("role")=="assistant":
                completion=m.get("content","")
            else:
                # include role name minimally to preserve structure
                user_chunks.append(f"{m.get('role','user')}: {m.get('content','')}")
        prompt="\n".join(user_chunks).strip()
        return {"prompt":prompt, "completion":completion}
    elif "prompt" in sample and "completion" in sample:
        return {"prompt":sample["prompt"], "completion":sample["completion"]}
    else:
        # fallback: treat whole line as prompt, empty completion
        return {"prompt":json.dumps(sample, ensure_ascii=False), "completion":""}

def build_supervised_examples(lines: List[Dict[str,Any]]) -> List[Dict[str,str]]:
    out=[]
    for s in lines:
        qa=extract_qa(s)
        if qa["completion"]=="":
            # skip samples without a target
            continue
        out.append(qa)
    return out

def format_example(tokenizer, prompt:str, completion:str):
    # single-turn instruction style
    # We mark assistant part with special tokens so we can mask loss on the prompt.
    chat_in = f"<|user|>\n{prompt}\n<|assistant|>\n"
    full = chat_in + completion
    ids = tokenizer(full, truncation=True, max_length=args.cutoff_len).input_ids
    # labels: ignore the prompt tokens, train only on assistant segment
    prompt_ids = tokenizer(chat_in, truncation=True, max_length=args.cutoff_len).input_ids
    labels = [-100]*len(prompt_ids) + ids[len(prompt_ids):]
    return {"input_ids": ids, "labels": labels}

@dataclass
class ConstantLengthCollator:
    tokenizer: Any
    mlm: bool = False
    def __call__(self, batch):
        maxlen = max(len(x["input_ids"]) for x in batch)
        input_ids = []
        labels = []
        for ex in batch:
            pad_len = maxlen - len(ex["input_ids"])
            input_ids.append(ex["input_ids"] + [self.tokenizer.pad_token_id]*pad_len)
            labels.append(ex["labels"] + [-100]*pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor([[1]*len(x["input_ids"]) + [0]*(maxlen-len(x["input_ids"])) for x in batch], dtype=torch.long)
        }

if __name__=="__main__":
    args = parse_args()

    # Load base model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_kwargs = {}
    try:
        # 4-bit if bitsandbytes available and GPU
        import bitsandbytes as bnb  # noqa: F401
        quant_kwargs = dict(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16)
    except Exception:
        pass

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16 if torch.cuda.is_available() else torch.float32,
        **quant_kwargs
    )

    # Attach LoRA
    peft_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, task_type=TaskType.CAUSAL_LM,
        target_modules=None  # let PEFT pick common modules
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    # Load + prepare dataset
    lines = load_lines(args.train_jsonl)
    pairs = build_supervised_examples(lines)
    if len(pairs)==0:
        raise SystemExit(f"No usable samples in {args.train_jsonl}")

    ds = [{"input_ids":[], "labels":[]} for _ in pairs]
    ds = [format_example(tokenizer, p["prompt"], p["completion"]) for p in pairs]

    # simple split: 90/10
    n = len(ds)
    n_train = max(1, int(0.9*n))
    train_ds = ds[:n_train]
    val_ds   = ds[n_train:]

    collator = ConstantLengthCollator(tokenizer=tokenizer)

    args_hf = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1,args.batch_size),
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        fp16=torch.cuda.is_available() and not args.bf16,
        bf16=args.bf16 and torch.cuda.is_available(),
        logging_steps=10,
        save_steps=200,
        evaluation_strategy="steps",
        eval_steps=100,
        save_total_limit=2,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        report_to="none"
    )

    class MapDS(torch.utils.data.Dataset):
        def __init__(self, data): self.data=data
        def __len__(self): return len(self.data)
        def __getitem__(self, i): return self.data[i]

    trainer = Trainer(
        model=model,
        args=args_hf,
        train_dataset=MapDS(train_ds),
        eval_dataset=MapDS(val_ds),
        data_collator=collator
    )

    trainer.train()
    model.save_pretrained(os.path.join(args.output_dir, "adapter"))
    tokenizer.save_pretrained(args.output_dir)
    print(f"[DONE] Saved LoRA adapter to {os.path.join(args.output_dir,'adapter')}")
