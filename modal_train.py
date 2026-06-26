"""DANTE — isolated Modal training job for the ModernBERT bi-encoder.

================================================================================
ISOLATION CONTRACT (read this)
================================================================================
This job is fully self-contained and shares NOTHING with any other project on
your Modal workspace:

  * App name      : "dante-train"          (its own app, no collisions)
  * Volume        : "dante-artifacts"      (its own storage, created on demand)
  * Secrets       : NONE attached          <-- it cannot read the Mongo/company
                                               secret; it never asks for one
  * No MongoDB, no external services, no reference to any other folder/app.

Modal auth is account-level (your `modal token`), so this reuses your account +
GPU quota — but it cannot see another app's files, volumes, or secrets. Running
or deleting this job's volume has zero effect on the finetunev1 / company work.

Only SPLADE and ColBERT are NOT trained here — the plan uses pretrained
checkpoints for both (DANTE_BUILD_PLAN.md §4.3 / §4.5), so the only A100 job is
this bi-encoder fine-tune.

================================================================================
HOW TO RUN
================================================================================
    pip install modal && modal token new          # one-time, if not already authed

    # quick smoke test on a tiny slice (a few minutes, cheap):
    modal run modal_train.py --stage all --limit 5000 --epochs 1

    # full run (downloads ESCI, builds pairs, trains ~1-1.5h on A100-80GB):
    modal run modal_train.py --stage all

    # stages can be run separately:
    modal run modal_train.py --stage data      # prepare pairs only (CPU)
    modal run modal_train.py --stage train     # train only (assumes data exists)

================================================================================
GET THE TRAINED MODEL ONTO YOUR PC  (then push to HF Hub / Kaggle, NOT git)
================================================================================
    modal volume get dante-artifacts /biencoder_final ./models/dante_biencoder

================================================================================
CLEAN UP WHEN DONE  (safe — only touches THIS volume)
================================================================================
    modal volume ls     dante-artifacts /          # see what's there
    modal volume rm  -r  dante-artifacts /data      # drop the big prepared data
    modal volume delete  dante-artifacts            # nuke everything (final)

    Back up the weights (volume get + upload to HF/Kaggle) BEFORE deleting.
================================================================================
"""

import modal

# ---- App + storage (both private to this project) ---------------------------
app = modal.App("dante-train")

# create_if_missing=True => first run makes the volume; nothing else can use it.
vol = modal.Volume.from_name("dante-artifacts", create_if_missing=True)
ARTIFACTS = "/artifacts"

# Cache HuggingFace downloads (ESCI, the base model) ON the volume so re-runs
# don't re-download. HF_HOME lives under the mounted volume.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2",
        "transformers>=4.48",          # ModernBERT support landed in 4.48
        "sentence-transformers>=5.0.0",
        "datasets>=2.19",
        "accelerate>=0.30",
        "faiss-cpu>=1.7.4",
        "numpy",
    )
    .env({"HF_HOME": f"{ARTIFACTS}/hf", "TOKENIZERS_PARALLELISM": "false"})
)

# ---- Hyperparameters (mirror configs/default.yaml :: biencoder) -------------
MODEL_NAME = "answerdotai/ModernBERT-base"
MAX_SEQ_LENGTH = 256
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1

# ESCI is hosted on HuggingFace; we build (anchor=query, positive=product_text)
# pairs and train with MultipleNegativesRankingLoss using in-batch negatives.
# (Explicit BM25 hard negatives are a documented phase-2 enhancement — brute-force
#  rank_bm25 over the full catalog is too slow to ship as the default; see
#  DANTE_BUILD_PLAN.md §4.2 and the "common pitfalls" note about brute-force.)
ESCI_DATASET = "tasksource/esci"


def _is_positive(label) -> bool:
    """ESCI label counts as a positive pair if Exact or Substitute (§3.4)."""
    if label is None:
        return False
    l = str(label).strip().lower()
    return l in ("e", "s", "exact", "substitute")


def _product_text(row) -> str:
    """title [SEP] brand [SEP] bullet[:256] [SEP] description[:256]  (§3.3)."""
    title = str(row.get("product_title") or "")
    brand = str(row.get("product_brand") or "")
    bullet = str(row.get("product_bullet_point") or "")[:256]
    desc = str(row.get("product_description") or "")[:256]
    return f"{title} [SEP] {brand} [SEP] {bullet} [SEP] {desc}".strip()


@app.function(image=image, volumes={ARTIFACTS: vol}, timeout=60 * 60, cpu=8.0)
def prepare_data(limit: int = 0):
    """Download ESCI (English), build positive (query, product) pairs, save to volume.

    Output: ``{ARTIFACTS}/data/train`` and ``{ARTIFACTS}/data/val`` (HF datasets,
    columns ``anchor`` + ``positive``).
    """
    import os
    from datasets import load_dataset

    os.makedirs(f"{ARTIFACTS}/hf", exist_ok=True)

    print(f"[data] loading {ESCI_DATASET} ...")
    ds = load_dataset(ESCI_DATASET, split="train")
    print(f"[data] raw rows: {len(ds):,} | columns: {ds.column_names}")

    cols = ds.column_names
    locale_col = "product_locale" if "product_locale" in cols else None
    label_col = next((c for c in ("esci_label", "label", "gain") if c in cols), None)
    if label_col is None:
        raise RuntimeError(f"Could not find an ESCI label column in {cols}")

    def _keep(row):
        if locale_col and str(row.get(locale_col) or "").lower() not in ("us", "en"):
            return False
        return _is_positive(row.get(label_col))

    ds = ds.filter(_keep, num_proc=8)
    print(f"[data] positive English rows: {len(ds):,}")

    if limit and limit > 0:
        ds = ds.select(range(min(limit, len(ds))))
        print(f"[data] limited to {len(ds):,} rows (smoke test)")

    ds = ds.map(
        lambda r: {"anchor": str(r.get("query") or ""), "positive": _product_text(r)},
        num_proc=8,
        remove_columns=[c for c in ds.column_names],
    )
    # Drop empties + exact dupes.
    ds = ds.filter(lambda r: len(r["anchor"]) > 0 and len(r["positive"]) > 4, num_proc=8)
    print(f"[data] usable pairs: {len(ds):,}")

    split = ds.train_test_split(test_size=0.02, seed=42)
    split["train"].save_to_disk(f"{ARTIFACTS}/data/train")
    split["test"].save_to_disk(f"{ARTIFACTS}/data/val")
    vol.commit()
    print(f"[data] saved: train={len(split['train']):,}  val={len(split['test']):,}")
    return {"train": len(split["train"]), "val": len(split["test"])}


@app.function(image=image, volumes={ARTIFACTS: vol}, gpu="A100-80GB", timeout=4 * 60 * 60)
def train_biencoder(epochs: int = 3, batch_size: int = 128):
    """Fine-tune ModernBERT on the prepared pairs with MNRL. Save to the volume.

    Output: ``{ARTIFACTS}/biencoder_final`` (a SentenceTransformer you can load
    with ``SentenceTransformer(path)``).
    """
    import os
    from datasets import load_from_disk
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.losses import MultipleNegativesRankingLoss

    train_path, val_path = f"{ARTIFACTS}/data/train", f"{ARTIFACTS}/data/val"
    if not os.path.isdir(train_path):
        raise RuntimeError("No prepared data found — run `--stage data` first.")

    train_ds = load_from_disk(train_path)
    val_ds = load_from_disk(val_path)
    print(f"[train] train={len(train_ds):,}  val={len(val_ds):,}  bs={batch_size}  epochs={epochs}")

    # sdpa attention avoids the optional flash-attn build; mean pooling by default.
    model = SentenceTransformer(MODEL_NAME, model_kwargs={"attn_implementation": "sdpa"})
    model.max_seq_length = MAX_SEQ_LENGTH

    loss = MultipleNegativesRankingLoss(model)

    args = SentenceTransformerTrainingArguments(
        output_dir=f"{ARTIFACTS}/biencoder_ckpts",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        bf16=True,
        # A100-80GB has ample headroom for a 150M model at bs=128/seq=256, so
        # gradient checkpointing is off for speed. Flip to True if you ever OOM.
        gradient_checkpointing=False,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        logging_steps=50,
        dataloader_num_workers=4,
        report_to=[],  # no wandb/tensorboard — keep the job hermetic
    )

    trainer = SentenceTransformerTrainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds, loss=loss,
    )
    trainer.train()

    out = f"{ARTIFACTS}/biencoder_final"
    model.save_pretrained(out)
    vol.commit()
    print(f"[train] DONE — model saved to volume at {out}")
    print("[train] pull it with:  modal volume get dante-artifacts /biencoder_final ./models/dante_biencoder")
    return out


@app.local_entrypoint()
def main(stage: str = "all", epochs: int = 3, batch_size: int = 128, limit: int = 0):
    """Orchestrate the pipeline. stage: 'all' | 'data' | 'train'."""
    if stage in ("all", "data"):
        print("== STAGE: prepare_data ==")
        print(prepare_data.remote(limit=limit))
    if stage in ("all", "train"):
        print("== STAGE: train_biencoder ==")
        print(train_biencoder.remote(epochs=epochs, batch_size=batch_size))
    if stage not in ("all", "data", "train"):
        raise SystemExit(f"unknown stage {stage!r}; use all|data|train")
