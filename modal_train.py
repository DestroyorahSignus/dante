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
        # Pinned to the exact versions validated on Modal 2026-06-28 (data + train
        # smoke both passed) for reproducibility — see DANTE_BUILD_PLAN.md §9.
        "torch==2.12.1",
        "transformers==5.12.1",        # ModernBERT works; warmup_ratio warns but is honored
        "sentence-transformers==5.6.0",
        "datasets==5.0.0",
        "accelerate==1.14.0",
        "faiss-cpu==1.14.3",
        "numpy==2.4.6",
        "pandas==3.0.3",               # data prep: split-by-query, qrels/catalog build
        "wandb",                       # experiment tracking (see train_biencoder)
        # --- index/ablation/preflight stages (DANTE_BUILD_PLAN §4/§5) ---
        "rank-bm25==0.2.2",            # BM25 lexical leg (§4.1)
        "scipy==1.16.0",               # CSR sparse matmul for SPLADE scoring (R3)
        # pylate = modern ST-native ColBERT for the reranker (§4.5). It is left
        # UNPINNED here on purpose: if pip would force-downgrade the pinned
        # torch/transformers above to satisfy pylate, the colbert_reranker's
        # graceful identity-fallback keeps the ablation running, so the image must
        # NOT break. If you see a downgrade in the build log, either pin a pylate
        # release compatible with torch 2.12 / transformers 5.12 or drop this line
        # and rely on the fallback (the "+ ColBERT" row then == the fused row).
        "pylate",
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


@app.function(
    image=image,
    volumes={ARTIFACTS: vol},
    gpu="A100-80GB",
    timeout=4 * 60 * 60,
    # W&B logging only. This secret holds ONLY WANDB_API_KEY — it does NOT expose the
    # company Mongo secret. Runs log to the WANDB_PROJECT set below.
    secrets=[modal.Secret.from_name("dante-wandb")],
)
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

    # W&B: the dante-wandb secret provides WANDB_API_KEY; log to a namespaced project
    # (company entity, but kept under its own project so portfolio runs are separable).
    os.environ.setdefault("WANDB_PROJECT", "dante-portfolio")

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
        report_to=["wandb"],  # log to W&B (dante-portfolio); key via dante-wandb secret
        run_name=f"dante-biencoder-e{epochs}-bs{batch_size}",
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


# A default config that mirrors configs/default.yaml but points at the volume.
# (The package code reads nested dicts, so this is passed straight through.)
def _default_config() -> dict:
    return {
        "biencoder": {"path": f"{ARTIFACTS}/biencoder_final"},
        "splade": {"model": "naver/splade-cocondenser-ensembledistil", "max_length": 256},
        "colbert": {"model": "answerdotai/answerai-colbert-small-v1"},
        "serving": {
            "catalog_path": f"{ARTIFACTS}/data/catalog.parquet",
            "index_dir": f"{ARTIFACTS}/index",
            "queries_path": f"{ARTIFACTS}/data/queries.json",
            "rrf_k": 60, "top_n": 200, "leg_top_k": 1000,
        },
        "eval": {"ks": [10, 50, 100, 200], "max_queries": 2000, "seed": 42,
                 "results_path": f"{ARTIFACTS}/ablation_results.json",
                 "queries_path": f"{ARTIFACTS}/data/queries.json"},
    }


@app.function(image=image, volumes={ARTIFACTS: vol}, gpu="A100-80GB", timeout=2 * 60 * 60)
def build_index():
    """Build dense FAISS + BM25 + SPLADE-CSR indices → ``/artifacts/index/`` (§4.6)."""
    from dante.serving.index_builder import build_indices

    paths = build_indices(_default_config())
    vol.commit()
    print(f"[index] committed: {paths}")
    return paths


@app.function(image=image, volumes={ARTIFACTS: vol}, gpu="A100-80GB", timeout=4 * 60 * 60)
def run_ablation():
    """Load the index + qrels + queries → run the 7-config ablation (§5.2)."""
    import json

    from dante.eval.evaluate import run_all_ablations
    from dante.serving.search_engine import DanteSearchEngine

    cfg = _default_config()
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
def preflight(n: int = 2000):
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

    cfg = _default_config()
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


@app.local_entrypoint()
def main(stage: str = "all", epochs: int = 3, batch_size: int = 128, limit: int = 0):
    """Orchestrate the pipeline.

    stage: 'all' | 'data' | 'train' | 'index' | 'ablation' | 'preflight'.
    ('all' runs data + train; index/ablation/preflight are run explicitly so the
     A100 index/eval passes don't fire on every smoke.)
    """
    valid = ("all", "data", "train", "index", "ablation", "preflight")
    if stage not in valid:
        raise SystemExit(f"unknown stage {stage!r}; use {'|'.join(valid)}")
    if stage in ("all", "data"):
        print("== STAGE: prepare_data ==")
        print(prepare_data.remote(limit=limit))
    if stage in ("all", "train"):
        print("== STAGE: train_biencoder ==")
        print(train_biencoder.remote(epochs=epochs, batch_size=batch_size))
    if stage == "index":
        print("== STAGE: build_index ==")
        print(build_index.remote())
    if stage == "ablation":
        print("== STAGE: run_ablation ==")
        print(run_ablation.remote())
    if stage == "preflight":
        print("== STAGE: preflight ==")
        print(preflight.remote())
