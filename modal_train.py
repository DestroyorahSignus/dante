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
IF A STAGE FAILS (CONTINGENCIES)
================================================================================
GENERAL RULE: every stage is idempotent and writes to the volume, so the universal
fallback is "fix the cause and re-run just that `--stage`." Data is cached, the HF
download is cached on the volume, and the trainer checkpoints to
`/artifacts/biencoder_ckpts` every 500 steps.

  * Modal preemption / timeout during training
        → just re-run `modal run modal_train.py --stage train`. Prepared data is on
          the volume; resume training from the latest checkpoint dir if needed.
  * GPU OOM (rare on A100-80GB)
        → lower batch: `--stage train --batch_size 64` (then 32). As a last resort set
          gradient_checkpointing=True in train_biencoder() and/or drop max_seq_length.
  * ESCI download 404 / schema drift / labels not the expected words
        → verify columns first (see RISKS R1); fall back to the authoritative source
          `git clone https://github.com/amazon-science/esci-data` and load its parquet.
  * HF rate-limit / flaky network
        → set HUGGING_FACE_HUB_TOKEN in the env and re-run; the volume cache means you
          don't re-download what already landed.
  * Dependency / import failure (e.g. ModernBERT)
        → ensure transformers>=4.48 + sentence-transformers>=5 (already pinned in the
          image); the model loads with attn_implementation="sdpa" to avoid flash-attn.
  * "0 usable pairs" after filtering
        → the label/locale/small_version filter is too strict for this mirror; print
          ds.unique("esci_label") and relax `_is_positive` / the small_version gate.
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
        "pandas>=2.0",        # data prep: split-by-query, qrels/catalog build
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

# Graded relevance (§3.4) — labels are full WORDS on this mirror, not E/S/C/I.
# Positives for contrastive training are grade >= 2 (Exact, Substitute).
GRADE = {"exact": 3, "substitute": 2, "complement": 1, "irrelevant": 0,
         "e": 3, "s": 2, "c": 1, "i": 0}
POS_GRADE = 2  # Exact + Substitute count as positives


def _grade(label) -> int:
    return GRADE.get(str(label).strip().lower(), 0)


def _product_text(row) -> str:
    """title [SEP] brand [SEP] bullet[:256] [SEP] description[:256]  (§3.3)."""
    title = str(row.get("product_title") or "")
    brand = str(row.get("product_brand") or "")
    bullet = str(row.get("product_bullet_point") or "")[:256]
    desc = str(row.get("product_description") or "")[:256]
    return f"{title} [SEP] {brand} [SEP] {bullet} [SEP] {desc}".strip()


@app.function(image=image, volumes={ARTIFACTS: vol}, timeout=90 * 60, cpu=8.0, memory=32768)
def prepare_data(limit: int = 0, max_pos_per_query: int = 16, val_frac: float = 0.01):
    """Build leakage-free DANTE train/eval data from ESCI and save to the volume.

    Produces (under {ARTIFACTS}/data/):
      train/            HF dataset [anchor, positive]   — MNRL training pairs
      val/              HF dataset [anchor, positive]   — eval-loss monitor (query-disjoint)
      catalog.parquet   [product_id, product_text]      — retrieval pool for recall@K
      qrels.json        {query_id: {product_id: grade}} — graded (Exact=3..Irrelevant=0)
      queries.json      {query_id: query_text}          — TEST queries
      stats.json        counts + the leakage assertion result

    Correctness properties this fixes:
      * Split is BY QUERY (official `split` column if the mirror has it, else a stable
        hash of query_id) — no query appears in both train and test (NO leakage).
      * Eval keeps ALL graded labels + a full catalog, so recall@K / nDCG are honest
        (retrieve against the whole catalog, not just the positives).
      * Positives (grade>=2) are deduped and capped per query so a few prolific queries
        don't dominate the contrastive batch.
    """
    import hashlib
    import json
    import os

    import pandas as pd
    from datasets import Dataset, load_dataset

    os.makedirs(f"{ARTIFACTS}/data", exist_ok=True)
    os.makedirs(f"{ARTIFACTS}/hf", exist_ok=True)

    print(f"[data] loading {ESCI_DATASET} ...")
    ds = load_dataset(ESCI_DATASET, split="train")
    cols = ds.column_names
    print(f"[data] raw rows: {len(ds):,} | columns: {cols}")

    locale_col = "product_locale" if "product_locale" in cols else None
    label_col = next((c for c in ("esci_label", "label", "gain") if c in cols), None)
    if label_col is None:
        raise RuntimeError(f"Could not find an ESCI label column in {cols}")
    qid_col = "query_id" if "query_id" in cols else None
    pid_col = "product_id" if "product_id" in cols else None
    has_small = "small_version" in cols
    has_split = "split" in cols  # official ESCI query-level split, if the mirror kept it

    # Base filter: reduced set + US, but KEEP ALL LABELS (eval needs negatives too).
    def _keep(r):
        if has_small and not r.get("small_version"):
            return False
        if locale_col and str(r.get(locale_col) or "").lower() != "us":
            return False
        return bool(r.get("query"))

    ds = ds.filter(_keep, num_proc=8)
    print(f"[data] US reduced-set rows (all labels): {len(ds):,}")
    if limit and limit > 0:
        ds = ds.select(range(min(limit, len(ds))))
        print(f"[data] limited to {len(ds):,} rows (smoke test)")

    # Project to needed columns and assign a WHOLE-QUERY train/test flag.
    def _project(r):
        qid = str(r.get(qid_col) if qid_col else (r.get("query") or ""))
        if has_split and r.get("split") is not None:
            is_test = str(r.get("split")).strip().lower() == "test"
        else:  # deterministic ~10% hash split — stable across runs, whole-query
            h = int(hashlib.md5(qid.encode("utf-8")).hexdigest(), 16)
            is_test = (h % 10) == 0
        return {
            "query": str(r.get("query") or ""),
            "query_id": qid,
            "product_id": str(r.get(pid_col) or ""),
            "product_text": _product_text(r),
            "grade": _grade(r.get(label_col)),
            "is_test": is_test,
        }

    ds = ds.map(_project, num_proc=8, remove_columns=cols)
    ds = ds.filter(
        lambda r: len(r["query"]) > 0 and len(r["product_id"]) > 0 and len(r["product_text"]) > 4,
        num_proc=8,
    )
    df = ds.to_pandas()

    # Catalog = every unique product = the retrieval pool for recall@K.
    catalog = df.drop_duplicates("product_id")[["product_id", "product_text"]]
    catalog.to_parquet(f"{ARTIFACTS}/data/catalog.parquet", index=False)

    train_df = df[~df["is_test"]]
    test_df = df[df["is_test"]]

    # Training positives: Exact+Substitute, deduped, capped per query.
    pos = (train_df[train_df["grade"] >= POS_GRADE]
           [["query_id", "query", "product_text"]]
           .drop_duplicates(["query_id", "product_text"]))
    pos = pos.groupby("query_id", group_keys=False).head(max_pos_per_query)
    pos = pos.rename(columns={"query": "anchor", "product_text": "positive"})

    # Query-disjoint val holdout (eval-loss only; the REAL eval is the test qrels).
    uniq_q = pos["query_id"].drop_duplicates()
    n_val_q = max(1, int(len(uniq_q) * val_frac)) if len(uniq_q) else 0
    val_qids = set(uniq_q.sample(n=n_val_q, random_state=42)) if n_val_q else set()
    val_pairs = pos[pos["query_id"].isin(val_qids)][["anchor", "positive"]]
    trn_pairs = pos[~pos["query_id"].isin(val_qids)][["anchor", "positive"]]

    Dataset.from_pandas(trn_pairs, preserve_index=False).save_to_disk(f"{ARTIFACTS}/data/train")
    Dataset.from_pandas(val_pairs, preserve_index=False).save_to_disk(f"{ARTIFACTS}/data/val")

    # Graded eval artifacts from the TEST split.
    qrels: dict = {}
    queries: dict = {}
    for row in test_df.itertuples(index=False):
        qrels.setdefault(row.query_id, {})[row.product_id] = int(row.grade)
        queries[row.query_id] = row.query
    with open(f"{ARTIFACTS}/data/qrels.json", "w") as f:
        json.dump(qrels, f)
    with open(f"{ARTIFACTS}/data/queries.json", "w") as f:
        json.dump(queries, f)

    # Leakage guard: train and test query sets MUST be disjoint.
    leak = set(train_df["query_id"].unique()) & set(test_df["query_id"].unique())
    assert not leak, f"QUERY LEAKAGE: {len(leak)} queries in both splits (e.g. {list(leak)[:3]})"

    stats = {
        "train_pairs": int(len(trn_pairs)),
        "val_pairs": int(len(val_pairs)),
        "catalog_products": int(len(catalog)),
        "test_queries": int(len(queries)),
        "test_judgements": int(sum(len(v) for v in qrels.values())),
        "split_source": "official 'split' column" if has_split else "hash(query_id)%10",
        "leakage": len(leak),
    }
    with open(f"{ARTIFACTS}/data/stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    vol.commit()
    print(f"[data] {stats}")
    return stats


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
