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
        # transformers 4.x era stack — PINNED to the versions validated on Modal
        # 2026-06-28 (train + index + 7-config ablation incl. a working ColBERT all
        # passed). ColBERT (answerai-colbert-small-v1 via rerankers) relies on
        # transformers-4.x internals REMOVED in 5.x (`generate_model_card`,
        # `all_tied_weights_keys`) → on 5.12 the reranker silently no-ops. ModernBERT +
        # MNRL train fine on 4.x and warmup_ratio is native (no deprecation).
        "torch==2.12.1",
        "transformers==4.57.6",        # 4.x: ModernBERT works AND ColBERT loads
        "sentence-transformers==4.1.0",
        "datasets==5.0.0",
        "accelerate==1.14.0",
        "faiss-cpu==1.14.3",
        "numpy==2.2.6",
        "pandas==3.0.3",               # data prep: split-by-query, qrels/catalog build
        "wandb==0.28.0",               # experiment tracking (see train_biencoder)
        # --- index/ablation/preflight stages (DANTE_BUILD_PLAN §4/§5) ---
        "rank-bm25==0.2.2",            # BM25 lexical leg (§4.1)
        "scipy==1.17.1",               # CSR sparse matmul for SPLADE scoring (R3)
        # ColBERT reranker (§4.5) via AnswerDotAI's `rerankers` (answerai-colbert-small-v1);
        # colbert_reranker keeps a graceful identity fallback so the ablation never crashes.
        "rerankers[transformers]==0.10.0",
    )
    .env({"HF_HOME": f"{ARTIFACTS}/hf", "TOKENIZERS_PARALLELISM": "false"})
    # Ship the dante/ source package so the index/ablation/preflight stages can
    # `from dante.serving... import ...`. modal_train.py sits at the repo root
    # alongside the dante/ package dir.
    .add_local_python_source("dante")
)

# ---- Hyperparameters (mirror configs/default.yaml :: biencoder) -------------
MODEL_NAME = "answerdotai/ModernBERT-base"
MAX_SEQ_LENGTH = 256
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1

# ESCI is hosted on HuggingFace; we build (anchor=query, positive=product_text)
# pairs and train with MultipleNegativesRankingLoss using in-batch negatives.
# (v0.2: the `mine` stage adds DENSE hard negatives per §4.2 — batched FAISS via
#  sentence_transformers.util.mine_hard_negatives, never brute-force rank_bm25 —
#  writing data/train_hn n-tuples that MNRL consumes natively via --train_dir.)
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


@app.function(
    image=image,
    volumes={ARTIFACTS: vol},
    gpu="A100-80GB",
    timeout=4 * 60 * 60,
    # W&B logging only. This secret holds ONLY WANDB_API_KEY — it does NOT expose the
    # company Mongo secret. Runs log to the WANDB_PROJECT set below.
    secrets=[modal.Secret.from_name("dante-wandb")],
)
def train_biencoder(epochs: int = 3, batch_size: int = 128,
                    train_dir: str = "data/train",
                    output_name: str = "biencoder_final",
                    base_model: str = MODEL_NAME):
    """Fine-tune a base encoder on the prepared pairs with MNRL. Save to the volume.

    Args:
        epochs / batch_size: standard hyperparameters (configs/default.yaml).
        base_model: HF checkpoint to fine-tune. Default ``answerdotai/ModernBERT-base``
            (v0.1 baseline). The §4.2 fallback ``BAAI/bge-base-en-v1.5`` also works —
            bge models encode raw text fine without instruction prefixes for this
            symmetric-ish product-search setup, so no prompt plumbing is needed.
        train_dir: volume-relative training dataset dir. Default ``data/train``
            = the v0.1 (anchor, positive) pairs (baseline reproducible). Pass
            ``data/train_hn`` to train on the mined hard-negative n-tuples —
            MNRL in ST 4.x natively treats every column after (anchor, positive),
            i.e. negative_1..negative_n, as explicit hard negatives on top of the
            in-batch ones.
        output_name: volume-relative output dir for the final model. Default
            ``biencoder_final`` (v0.1 path); pass e.g. ``biencoder_v2`` so a
            hard-negative run never clobbers the v0.1 weights.

    Output: ``{ARTIFACTS}/{output_name}`` (a SentenceTransformer you can load
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

    # W&B: the dante-wandb secret provides WANDB_API_KEY; log to a namespaced project
    # (company entity, but kept under its own project so portfolio runs are separable).
    os.environ.setdefault("WANDB_PROJECT", "dante-portfolio")

    train_path, val_path = f"{ARTIFACTS}/{train_dir}", f"{ARTIFACTS}/data/val"
    if not os.path.isdir(train_path):
        raise RuntimeError(f"No prepared data found at {train_path} — run "
                           "`--stage data` (and `--stage mine` for train_hn) first.")

    train_ds = load_from_disk(train_path)
    val_ds = load_from_disk(val_path)
    print(f"[train] train={len(train_ds):,} ({train_dir})  val={len(val_ds):,}  "
          f"bs={batch_size}  epochs={epochs}  base={base_model}  -> {output_name}")
    print(f"[train] train columns: {train_ds.column_names}")

    # sdpa attention avoids the optional flash-attn build; mean pooling by default.
    model = SentenceTransformer(base_model, model_kwargs={"attn_implementation": "sdpa"})
    model.max_seq_length = MAX_SEQ_LENGTH

    loss = MultipleNegativesRankingLoss(model)

    # Keep the historical ckpt dir for the default run (baseline reproducible);
    # non-default output names get their own ckpt dir so runs never mix.
    ckpt_dir = (f"{ARTIFACTS}/biencoder_ckpts" if output_name == "biencoder_final"
                else f"{ARTIFACTS}/{output_name}_ckpts")
    args = SentenceTransformerTrainingArguments(
        output_dir=ckpt_dir,
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
        report_to=["wandb"],  # log to W&B (dante-portfolio); key via dante-wandb secret
        run_name=(f"dante-biencoder-e{epochs}-bs{batch_size}"
                  + ("" if output_name == "biencoder_final" else f"-{output_name}")),
    )

    trainer = SentenceTransformerTrainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds, loss=loss,
    )
    trainer.train()

    out = f"{ARTIFACTS}/{output_name}"
    model.save_pretrained(out)
    vol.commit()
    print(f"[train] DONE — model saved to volume at {out}")
    print(f"[train] pull it with:  modal volume get dante-artifacts /{output_name} ./models/dante_biencoder")
    return out


@app.function(image=image, volumes={ARTIFACTS: vol}, gpu="A100-80GB", timeout=2 * 60 * 60)
def mine(num_negatives: int = 4, range_min: int = 1, range_max: int = 200,
         batch_size: int = 1024, output_format: str = "triplet"):
    """Mine DENSE hard negatives with the v0.1 bi-encoder (BUILD_PLAN §4.2, dense-first).

    Motivation (v0.1 ablation, 2,000 test queries): the Dense leg is the WEAKEST
    (R@200 0.627) because it was trained with in-batch negatives only. This stage
    mines semantically-confusable negatives from the FULL catalog with the trained
    v0.1 model — the batched-FAISS path §4.2 mandates (never brute-force rank_bm25).

    Reads : {ARTIFACTS}/biencoder_final, {ARTIFACTS}/data/train (anchor/positive),
            {ARTIFACTS}/data/catalog.parquet (full 351,961-product corpus).
    Writes: {ARTIFACTS}/data/train_hn — HF dataset with columns
            anchor, positive, negative_1..negative_{num_negatives}
            (output_format="n-tuple"; MNRL consumes the extra columns natively).
            {ARTIFACTS}/data/train is NOT touched (v0.1 stays reproducible).

    False-negative guards (§4.2 / RISKS R8 — do not mine unlabeled positives):
      * mine_hard_negatives itself never returns a row's OWN positive.
      * A query's OTHER labeled positives sit in the corpus too — we do not
        blocklist them explicitly; instead range_min=1 skips the top hit and the
        zero margin (absolute_margin/margin=0.0) drops any candidate scoring >=
        the row's positive, which is exactly the §4.2 recipe for keeping likely
        (unlabeled or other-labeled) positives out of the negative set.
    """
    import inspect
    import os

    import pandas as pd
    from datasets import load_from_disk
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import mine_hard_negatives

    train_path = f"{ARTIFACTS}/data/train"
    catalog_path = f"{ARTIFACTS}/data/catalog.parquet"
    model_path = f"{ARTIFACTS}/biencoder_final"
    for p in (train_path, catalog_path, model_path):
        if not os.path.exists(p):
            raise RuntimeError(f"Missing volume artifact {p} — run data/train stages first.")

    train_ds = load_from_disk(train_path)
    catalog = pd.read_parquet(catalog_path)
    corpus_texts = catalog["product_text"].astype(str).tolist()
    model = SentenceTransformer(model_path, model_kwargs={"attn_implementation": "sdpa"})
    print(f"[mine] pairs={len(train_ds):,}  corpus={len(corpus_texts):,}  "
          f"num_negatives={num_negatives}  range=[{range_min},{range_max}]")

    # Build kwargs defensively: ST versions rename things (margin -> absolute_margin/
    # relative_margin in 4.x; corpus= exists in 4.1.0). Inspect the runtime signature
    # and pass only what it supports, printing anything we had to drop.
    supported = set(inspect.signature(mine_hard_negatives).parameters)
    kwargs = {
        # negatives come from the WHOLE catalog, not just the positives' texts
        "corpus": corpus_texts,
        "num_negatives": num_negatives,
        "range_min": range_min,     # skip the top hit (likely an unlabeled positive)
        "range_max": range_max,     # mine within ranks [range_min, range_max]
        "sampling_strategy": "random",  # random within the range → diversity (§4.2)
        "batch_size": batch_size,   # A100-80GB encode batch
        # "triplet" (default) emits one (anchor, positive, negative) ROW per mined
        # negative → keeps every anchor that yields >=1 negative (n-tuple DROPS any
        # anchor that can't fill all num_negatives slots — that cost us 83% of rows
        # in the first pass: 41,213/244,179). MNRL consumes triplets natively.
        "output_format": output_format,
        "use_faiss": True,          # batched ANN — the whole point of §4.2
        "verbose": True,
    }
    # margin guard, whatever this ST version calls it (candidate sim must be
    # strictly below the positive's sim).
    if "absolute_margin" in supported:
        kwargs["absolute_margin"] = 0.0
    elif "margin" in supported:
        kwargs["margin"] = 0.0
    else:
        print("[mine] NOTE: no margin kwarg in this ST version — relying on range_min only")
    if "corpus" not in supported:
        # Fallback: mine within the positives' texts only (weaker corpus; §4.2 note).
        print("[mine] NOTE: this ST version's mine_hard_negatives has no corpus= — "
              "falling back to mining within the training positives only")
    dropped = sorted(set(kwargs) - supported)
    if dropped:
        print(f"[mine] NOTE: dropping unsupported kwargs for this ST version: {dropped}")
    kwargs = {k: v for k, v in kwargs.items() if k in supported}
    print(f"[mine] mine_hard_negatives kwargs: "
          f"{ {k: (f'<{len(v):,} texts>' if k == 'corpus' else v) for k, v in kwargs.items()} }")

    # datasets>=4 returns a lazy `Column` from ds["col"]; ST 4.1's mine_hard_negatives
    # calls .copy() on it (it expected the old list return) -> AttributeError. Patch the
    # RUNTIME column type (version-proof: taken from the actual object) with a .copy()
    # that materializes a plain list — surgical shim, no image/datasets downgrade.
    _col_t = type(train_ds["anchor"])
    if not isinstance(train_ds["anchor"], list) and not hasattr(_col_t, "copy"):
        _col_t.copy = lambda self: list(self)
        print(f"[mine] shimmed {_col_t.__name__}.copy() for ST-4.1 compat (datasets>=4 lazy Column)")

    mined = mine_hard_negatives(train_ds, model, **kwargs)

    neg_cols = [c for c in mined.column_names if c.startswith("negative")]
    out = f"{ARTIFACTS}/data/train_hn"
    mined.save_to_disk(out)
    vol.commit()
    stats = {"rows": len(mined), "columns": mined.column_names,
             "negatives_per_row": len(neg_cols),
             "input_pairs": len(train_ds)}
    print(f"[mine] DONE — saved {out}  {stats}")
    return stats


# A default config that mirrors configs/default.yaml but points at the volume.
# (The package code reads nested dicts, so this is passed straight through.)
# model_dir / index_dir / results_name are volume-relative overrides so v0.2 runs
# (e.g. biencoder_v2 + index_v2 + ablation_results_v2.json) never clobber v0.1.
def _default_config(model_dir: str = "biencoder_final", index_dir: str = "index",
                    results_name: str = "ablation_results.json") -> dict:
    return {
        "biencoder": {"path": f"{ARTIFACTS}/{model_dir}"},
        "splade": {"model": "opensearch-project/opensearch-neural-sparse-encoding-v2-distill", "max_length": 256},
        "colbert": {"model": "answerdotai/answerai-colbert-small-v1"},
        "serving": {
            "catalog_path": f"{ARTIFACTS}/data/catalog.parquet",
            "index_dir": f"{ARTIFACTS}/{index_dir}",
            "queries_path": f"{ARTIFACTS}/data/queries.json",
            "rrf_k": 60, "top_n": 200, "leg_top_k": 1000,
        },
        "eval": {"ks": [10, 50, 100, 200], "max_queries": 2000, "seed": 42,
                 "results_path": f"{ARTIFACTS}/{results_name}",
                 "queries_path": f"{ARTIFACTS}/data/queries.json"},
    }


@app.function(image=image, volumes={ARTIFACTS: vol}, gpu="A100-80GB", timeout=2 * 60 * 60)
def build_index(model_dir: str = "biencoder_final", index_dir: str = "index"):
    """Build dense FAISS + BM25 + SPLADE-CSR indices → ``{ARTIFACTS}/{index_dir}/`` (§4.6)."""
    from dante.serving.index_builder import build_indices

    paths = build_indices(_default_config(model_dir=model_dir, index_dir=index_dir))
    vol.commit()
    print(f"[index] committed: {paths}")
    return paths


@app.function(image=image, volumes={ARTIFACTS: vol}, gpu="A100-80GB", timeout=4 * 60 * 60)
def run_ablation(model_dir: str = "biencoder_final", index_dir: str = "index",
                 results_name: str = "ablation_results.json"):
    """Load the index + qrels + queries → run the 10-config ablation (§5.2)."""
    import json

    from dante.eval.evaluate import run_all_ablations
    from dante.serving.search_engine import DanteSearchEngine

    cfg = _default_config(model_dir=model_dir, index_dir=index_dir,
                          results_name=results_name)
    with open(f"{ARTIFACTS}/data/qrels.json") as f:
        qrels = json.load(f)
    with open(f"{ARTIFACTS}/data/queries.json") as f:
        queries = json.load(f)

    engine = DanteSearchEngine(cfg)
    out = run_all_ablations(
        engine, queries, qrels,
        ks=tuple(cfg["eval"]["ks"]),
        max_queries=cfg["eval"]["max_queries"],
        seed=cfg["eval"]["seed"],
    )
    with open(cfg["eval"]["results_path"], "w") as f:
        json.dump({"results": out["results"], "n_queries": out["n_queries"]}, f, indent=2)
    vol.commit()
    print(f"[ablation] wrote {cfg['eval']['results_path']}")
    return out["results"]


@app.function(image=image, volumes={ARTIFACTS: vol}, gpu="A100-80GB", timeout=60 * 60)
def preflight(n: int = 2000, model_dir: str = "biencoder_final", index_dir: str = "index"):
    """GPU sanity on the volume artifacts BEFORE trusting real-query numbers.

    1. FAISS self-test: encode N catalog docs, query with a doc's own text → expect
       that doc to come back rank-1 (validates encode+normalize+IndexFlatIP, R7).
    2. Retriever ceiling: query the FULL dense index with each product's own
       product_text → report rank-1 / rank-10 self-hit rate (the encoder/index
       ceiling; if low, real-query recall can't be trusted — fix first).
    3. SPLADE expansion sanity: a query expands into >0 sensible terms.
    """
    import json

    import faiss
    import numpy as np
    import pandas as pd
    from sentence_transformers import SentenceTransformer

    from dante.models.biencoder import build_dense_index, dense_search
    from dante.models.splade import SpladeEncoder, visualize_expansion

    cfg = _default_config(model_dir=model_dir, index_dir=index_dir)
    catalog = pd.read_parquet(cfg["serving"]["catalog_path"])
    ids = catalog["product_id"].astype(str).tolist()
    texts = catalog["product_text"].astype(str).tolist()
    model = SentenceTransformer(cfg["biencoder"]["path"])

    # --- 1. FAISS self-test on a small slice ---
    m = min(n, len(ids))
    sub_ids, sub_texts = ids[:m], texts[:m]
    sub_index, _ = build_dense_index(model, sub_ids, sub_texts)
    # unit-norm assert on a fresh encode (R7)
    emb = model.encode(sub_texts[:5], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(emb)
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-4), "embeddings not unit-norm"
    self_hits = sum(
        1 for i in range(m)
        if dense_search(model, sub_index, sub_ids, sub_texts[i], top_k=1)[0][0] == sub_ids[i]
    )
    faiss_selftest = self_hits / m

    # --- 2. Retriever ceiling against the FULL dense index ---
    full_index, _ = build_dense_index(model, ids, texts)
    sample = list(range(len(ids)))
    if len(sample) > n:
        rng = np.random.default_rng(42)
        sample = rng.choice(len(ids), size=n, replace=False).tolist()
    r1 = r10 = 0
    for i in sample:
        hits = [pid for pid, _ in dense_search(model, full_index, ids, texts[i], top_k=10)]
        if hits and hits[0] == ids[i]:
            r1 += 1
        if ids[i] in hits:
            r10 += 1
    ceiling = {"rank1": r1 / len(sample), "rank10": r10 / len(sample), "n": len(sample)}

    # --- 3. SPLADE expansion sanity ---
    splade = SpladeEncoder(model_name=cfg["splade"]["model"])
    expansion = visualize_expansion("wireless bluetooth headphones", splade, top_k_terms=10)

    report = {
        "faiss_selftest_rank1": faiss_selftest,
        "ceiling": ceiling,
        "splade_expansion_terms": len(expansion),
        "splade_expansion_sample": expansion[:5],
    }
    print(f"[preflight] {json.dumps(report, indent=2)}")
    assert faiss_selftest > 0.95, f"FAISS self-test rank-1 too low: {faiss_selftest:.3f}"
    assert len(expansion) > 0, "SPLADE produced no expansion terms"
    return report


@app.function(image=image, volumes={ARTIFACTS: vol}, gpu="A100-80GB", timeout=2 * 60 * 60)
def eval_enrich():
    """Eval-enrichment (NO retrain): dim-truncation ablation + RRF-k sweep (§4.2 / §14).

    Reuses the EXISTING trained bi-encoder + the catalog/qrels/queries on the volume.
    Only the dense leg is rebuilt per truncation dim (slice → renormalize → IndexFlatIP);
    the RRF sweep re-fuses the same cached leg lists at several k values.

    (a) dim ablation — dims [768, 256, 128]: dense recall@{10,50,100,200} + nDCG@10.
    (b) rrf sweep    — k in [10, 30, 60, 100]: nDCG@10 + R@200.

    Uses the SAME eval.max_queries subsample + seed as run_ablation, so the 768-d dim
    row and the k=60 sweep row line up with the baseline ablation. Writes
    /artifacts/eval_enrich.json and prints two tables.
    """
    import json

    import pandas as pd
    from sentence_transformers import SentenceTransformer

    from dante.eval.evaluate import dim_truncation_ablation, rrf_k_sweep
    from dante.serving.search_engine import DanteSearchEngine

    cfg = _default_config()
    ks = tuple(cfg["eval"]["ks"])
    max_queries = cfg["eval"]["max_queries"]
    seed = cfg["eval"]["seed"]

    with open(f"{ARTIFACTS}/data/qrels.json") as f:
        qrels = json.load(f)
    with open(f"{ARTIFACTS}/data/queries.json") as f:
        queries = json.load(f)

    # (a) Dimension-truncation ablation — needs only the trained encoder + catalog.
    catalog = pd.read_parquet(cfg["serving"]["catalog_path"])
    cat_ids = catalog["product_id"].astype(str).tolist()
    cat_texts = catalog["product_text"].astype(str).tolist()
    model = SentenceTransformer(cfg["biencoder"]["path"])
    dim_out = dim_truncation_ablation(
        model, cat_ids, cat_texts, queries, qrels,
        dims=(768, 256, 128), ks=ks,
        max_queries=max_queries, seed=seed,
        leg_top_k=cfg["serving"]["leg_top_k"],
    )

    # (b) RRF-k sweep — reuse the full engine (loads the existing 3-leg indices).
    engine = DanteSearchEngine(cfg)
    rrf_out = rrf_k_sweep(
        engine, queries, qrels,
        k_values=(10, 30, 60, 100), legs=("dense", "bm25", "splade"),
        ks=ks, max_queries=max_queries, seed=seed,
    )

    out = {
        "dim_ablation": {
            "results": dim_out["results"], "full_dim": dim_out["full_dim"],
            "n_queries": dim_out["n_queries"],
        },
        "rrf_sweep": {
            "results": rrf_out["results"], "n_queries": rrf_out["n_queries"],
        },
        "config": {"ks": list(ks), "max_queries": max_queries, "seed": seed},
    }
    out_path = f"{ARTIFACTS}/eval_enrich.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    vol.commit()
    print(f"[eval_enrich] wrote {out_path}")
    print("\n=== DIM-TRUNCATION ABLATION ===\n" + dim_out["table"])
    print("\n=== RRF-k SWEEP ===\n" + rrf_out["table"])
    return out


@app.local_entrypoint()
def main(stage: str = "all", epochs: int = 3, batch_size: int = 128, limit: int = 0,
         train_dir: str = "data/train", output_name: str = "biencoder_final",
         base_model: str = MODEL_NAME,
         model_dir: str = "biencoder_final", index_dir: str = "index",
         results_name: str = "ablation_results.json", num_negatives: int = 4,
         mine_range_max: int = 200, mine_output_format: str = "triplet"):
    """Orchestrate the pipeline.

    stage: 'all' | 'data' | 'train' | 'mine' | 'index' | 'ablation' | 'preflight'
           | 'eval_enrich'.
    ('all' runs data + train; mine/index/ablation/preflight/eval_enrich are run
     explicitly so the A100 index/eval passes don't fire on every smoke.)

    v0.2 hard-negative flow (defaults keep v0.1 fully reproducible):
        modal run modal_train.py --stage mine
        modal run modal_train.py --stage train --train-dir data/train_hn --output-name biencoder_v2 --batch-size 64
        # parallel retrain candidate on the §4.2 fallback base:
        modal run modal_train.py --stage train --train-dir data/train_hn --output-name biencoder_v2_bge --base-model BAAI/bge-base-en-v1.5 --batch-size 64
        modal run modal_train.py --stage index --model-dir biencoder_v2 --index-dir index_v2
        modal run modal_train.py --stage preflight --model-dir biencoder_v2 --index-dir index_v2
        modal run modal_train.py --stage ablation --model-dir biencoder_v2 --index-dir index_v2 --results-name ablation_results_v2.json
    """
    valid = ("all", "data", "train", "mine", "index", "ablation", "preflight",
             "eval_enrich")
    if stage not in valid:
        raise SystemExit(f"unknown stage {stage!r}; use {'|'.join(valid)}")
    if stage in ("all", "data"):
        print("== STAGE: prepare_data ==")
        print(prepare_data.remote(limit=limit))
    if stage in ("all", "train"):
        print("== STAGE: train_biencoder ==")
        print(train_biencoder.remote(epochs=epochs, batch_size=batch_size,
                                     train_dir=train_dir, output_name=output_name,
                                     base_model=base_model))
    if stage == "mine":
        print("== STAGE: mine (dense hard negatives) ==")
        print(mine.remote(num_negatives=num_negatives, range_max=mine_range_max,
                          output_format=mine_output_format))
    if stage == "index":
        print("== STAGE: build_index ==")
        print(build_index.remote(model_dir=model_dir, index_dir=index_dir))
    if stage == "ablation":
        print("== STAGE: run_ablation ==")
        print(run_ablation.remote(model_dir=model_dir, index_dir=index_dir,
                                  results_name=results_name))
    if stage == "preflight":
        print("== STAGE: preflight ==")
        print(preflight.remote(model_dir=model_dir, index_dir=index_dir))
    if stage == "eval_enrich":
        print("== STAGE: eval_enrich ==")
        print(eval_enrich.remote())
