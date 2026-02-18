from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def batched(items: List[str], batch_size: int) -> Iterable[List[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate MedCPT-Article-Encoder embeddings for report CSV files and save JSONL outputs."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input CSV path with report column.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL file path.")
    parser.add_argument(
        "--model",
        type=str,
        default="ncbi/MedCPT-Article-Encoder",
        help="HF model name.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help=(
            "Pinned HF revision (commit hash or tag) for reproducible embeddings. "
            "Required unless --allow-unpinned is set."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Encoding batch size.")
    parser.add_argument("--max-length", type=int, default=512, help="Tokenizer truncation length.")
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize embedding vectors to unit length.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume appending to an existing output JSONL (skip already embedded rows).",
    )
    parser.add_argument(
        "--allow-unpinned",
        action="store_true",
        help="Allow running without --revision. Intended only for exploratory runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.revision and not args.allow_unpinned:
        raise ValueError(
            "Unpinned embedding model revision. Pass --revision <commit/tag> for reproducibility "
            "or use --allow-unpinned for exploratory runs."
        )

    df = pd.read_csv(args.input, dtype=str)

    required = {"id", "dataset", "report"}
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {args.input}: {missing}")

    reports = df["report"].fillna("").astype(str).tolist()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModel.from_pretrained(args.model, revision=args.revision).to(device)
    model.eval()
    embedding_revision = (
        args.revision
        or getattr(getattr(model, "config", None), "_commit_hash", None)
        or "unpinned"
    )
    pooling: Optional[str] = None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    start_index = 0
    mode = "w"
    if args.resume and args.output.exists():
        with args.output.open("r", encoding="utf-8") as existing:
            for line_no, line in enumerate(existing, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in existing output {args.output} line {line_no}: {exc.msg}"
                    ) from exc
                if start_index >= len(df):
                    raise ValueError(
                        f"Existing output has more rows than input ({start_index + 1} > {len(df)})."
                    )

                row = df.iloc[start_index]
                expected_dataset = str(row["dataset"]).strip()
                expected_id = str(row["id"]).strip()
                seen_dataset = str(payload.get("dataset", "")).strip()
                seen_id = str(payload.get("id", "")).strip()
                if seen_dataset != expected_dataset or seen_id != expected_id:
                    raise ValueError(
                        "Resume safety check failed at output line "
                        f"{line_no}: expected ({expected_dataset}, {expected_id}) but found "
                        f"({seen_dataset}, {seen_id})."
                    )

                seen_model = str(payload.get("embedding_model", "")).strip()
                if seen_model and seen_model != args.model:
                    raise ValueError(
                        f"Resume safety check failed: existing embedding_model={seen_model!r} "
                        f"does not match requested model={args.model!r}."
                    )

                start_index += 1

        mode = "a"
        if start_index > len(df):
            raise ValueError(
                f"Existing output has {start_index} rows, but input only has {len(df)} rows."
            )
        print(f"Resuming from row {start_index} of {len(df)}")

    with args.output.open(mode, encoding="utf-8") as handle:
        index = start_index
        for report_batch in batched(reports[start_index:], args.batch_size):
            encoded = tokenizer(
                report_batch,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            with torch.no_grad():
                outputs = model(**encoded)
                # MedCPT article encoder exposes pooler_output (CLS projection).
                if getattr(outputs, "pooler_output", None) is not None:
                    batch_pooling = "pooler_output"
                    emb = outputs.pooler_output
                else:
                    batch_pooling = "cls"
                    emb = outputs.last_hidden_state[:, 0, :]
                if pooling is None:
                    pooling = batch_pooling
                elif pooling != batch_pooling:
                    raise RuntimeError(
                        f"Inconsistent pooling strategy detected across batches: {pooling} vs {batch_pooling}"
                    )
                if args.normalize:
                    emb = F.normalize(emb, p=2, dim=1)
                emb_np = emb.detach().cpu().numpy()

            for vector in emb_np:
                row = df.iloc[index]
                payload = {
                    "dataset": str(row["dataset"]),
                    "id": str(row["id"]),
                    "report": str(row["report"]) if row["report"] is not None else "",
                    "embedding_model": args.model,
                    "embedding_revision": embedding_revision,
                    "tokenizer_model": args.model,
                    "tokenizer_revision": embedding_revision,
                    "embedding_pooling": pooling or "unknown",
                    "max_length": int(args.max_length),
                    "normalized": bool(args.normalize),
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "embedding": vector.astype(np.float32).tolist(),
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                index += 1

    print(f"Wrote {len(df) - start_index} rows to {args.output}")


if __name__ == "__main__":
    main()
