"""Precompute the UMT5-XXL text contexts of every task string in a LeRobot dataset for FastWAM.

FastWAM conditions both DiTs on a fixed-length (tokenizer_max_len x 4096) UMT5 embedding of
`prompt_template.format(task=...)`. The encoder is frozen and LIBERO has only 40 task strings, so the
contexts can be computed once here and looked up during training, evaluation and uncertainty scoring
(`policy.text_context_path`), letting `load_text_encoder: false` keep the 11 GB encoder off the GPU.
Encoding goes through `encode_prompt_with_encoder`, the same function `FastWAM.encode_prompt` uses,
so the table is bit-identical to on-the-fly encoding.

    python scripts/fastwam/precompute_task_contexts.py --repo-id HuggingFaceVLA/libero \
        --out outputs/pretrain/fastwam_text_contexts/libero.pt
"""

from __future__ import annotations

import argparse
import pathlib

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.fastwam.configuration_fastwam import FastWAMConfig
from lerobot.policies.fastwam.wan.components import build_wan_tokenizer, load_pretrained_wan_text_encoder
from lerobot.policies.fastwam.wan.modular import encode_prompt_with_encoder


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", action="append", required=True, help="dataset(s) whose task strings to encode")
    ap.add_argument("--root", default=None, help="local dataset root (single --repo-id only)")
    ap.add_argument("--extra-task", action="append", default=[], help="additional raw task strings")
    ap.add_argument("--out", required=True, help="output .pt file")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--prompt-template", default=FastWAMConfig.prompt_template)
    ap.add_argument("--text-encoder-model-id", default=FastWAMConfig.text_encoder_model_id)
    ap.add_argument("--tokenizer-model-id", default=FastWAMConfig.tokenizer_model_id)
    ap.add_argument("--tokenizer-max-len", type=int, default=FastWAMConfig.tokenizer_max_len)
    args = ap.parse_args()

    tasks: list[str] = []
    for repo_id in args.repo_id:
        meta = LeRobotDatasetMetadata(repo_id, root=args.root if len(args.repo_id) == 1 else None)
        names = [str(t) for t in meta.tasks.index]
        print(f"{repo_id}: {len(names)} task strings")
        tasks.extend(names)
    tasks.extend(args.extra_task)
    tasks = list(dict.fromkeys(tasks))  # dedupe, keep order
    prompts = [args.prompt_template.format(task=t) for t in tasks]

    dtype = torch.bfloat16
    tokenizer = build_wan_tokenizer(model_id=args.tokenizer_model_id, tokenizer_max_len=args.tokenizer_max_len)
    encoder = load_pretrained_wan_text_encoder(model_id=args.text_encoder_model_id, torch_dtype=dtype, device=args.device)
    contexts, masks = [], []
    for start in range(0, len(prompts), args.batch_size):
        ctx, mask = encode_prompt_with_encoder(encoder, tokenizer, prompts[start : start + args.batch_size], args.device)
        contexts.append(ctx.to("cpu", dtype))
        masks.append(mask.cpu())
    context = torch.cat(contexts)
    context_mask = torch.cat(masks)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "tasks": tasks,
            "prompts": prompts,
            "context": context,
            "context_mask": context_mask,
            "prompt_template": args.prompt_template,
            "text_encoder_model_id": args.text_encoder_model_id,
            "tokenizer_model_id": args.tokenizer_model_id,
            "tokenizer_max_len": args.tokenizer_max_len,
            "repo_ids": args.repo_id,
        },
        out,
    )
    print(
        f"wrote {out} ({out.stat().st_size / 2**20:.0f} MiB): {len(prompts)} prompts, "
        f"context {tuple(context.shape)} {context.dtype}, non-zero tokens per prompt "
        f"{[int((context[i].abs().sum(-1) > 0).sum()) for i in range(min(3, len(prompts)))]}..."
    )


if __name__ == "__main__":
    main()
