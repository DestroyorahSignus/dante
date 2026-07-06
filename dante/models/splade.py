"""SPLADE-v3 learned-sparse retrieval — DANTE_BUILD_PLAN.md §4.3.

``SpladeEncoder`` turns text into a sparse term-weight vector via the standard
SPLADE aggregation (``max`` over sequence of ``log1p(relu(logits))``). The catalog
index stores all doc vectors as ONE ``scipy.sparse`` CSR matrix and scores a query
with a single sparse matmul ``q @ doc_matrix.T`` (RISK R3) — never a per-doc Python
loop.

Model choice (``DEFAULT_MODEL``): ``opensearch-project/opensearch-neural-sparse-encoding-v2-distill``.
  * License: Apache-2.0 (COMMERCIAL-friendly), ungated — replaces the previous
    ``naver/splade-cocondenser-ensembledistil`` (CC-BY-NC-SA, non-commercial), which
    is unusable for a commercial portfolio.
  * Architecture: ``DistilBertForMaskedLM`` (a standard ``*ForMaskedLM``), so it loads
    via ``AutoModelForMaskedLM`` and the SPLADE ``max(log1p(relu(logits))*mask)``
    aggregation below applies UNCHANGED (verified against its config.json).
  * Quality: BEIR avg nDCG@10 ~0.528.
The model id is config-driven (``splade.model`` in configs/default.yaml); this default
is only the fallback when no config value is supplied.
"""
from __future__ import annotations

# Apache-2.0, ungated, DistilBertForMaskedLM, BEIR ~0.528. See module docstring.
DEFAULT_MODEL = "opensearch-project/opensearch-neural-sparse-encoding-v2-distill"


class SpladeEncoder:
    """Encode text into a sparse SPLADE vector and run CSR-matmul catalog search."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None,
                 max_length: int = 256) -> None:
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.eval()
        if device is None:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(device)
        self.vocab_size = int(self.model.config.vocab_size)

        # Populated by build_index().
        self.doc_ids: list[str] = []
        self._doc_matrix = None  # scipy.sparse.csr_matrix [num_docs, vocab_size]

    # -- core encoding -------------------------------------------------------
    def _encode_weights(self, text: str):
        """Return the dense vocab-size weight tensor for a single text (on CPU)."""
        import torch

        inputs = self.tokenizer(
            str(text), return_tensors="pt", truncation=True, max_length=self.max_length
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits  # [1, seq, vocab]
            mask = inputs["attention_mask"].unsqueeze(-1)  # [1, seq, 1]
            weighted = torch.log1p(torch.relu(logits)) * mask
            weights = torch.max(weighted, dim=1).values.squeeze(0)  # [vocab]
        return weights.detach().cpu()

    def encode(self, text: str) -> dict[str, float]:
        """Encode text into a sparse ``{term_string: weight}`` dict (interpretable)."""
        weights = self._encode_weights(text)
        nz = weights.nonzero().squeeze(-1)
        sparse: dict[str, float] = {}
        for idx in nz.tolist():
            sparse[self.tokenizer.decode([idx])] = float(weights[idx])
        return sparse

    def encode_sparse_row(self, text: str):
        """Encode text into a ``scipy.sparse`` CSR row over the token-id vocab."""
        from scipy import sparse as sp

        weights = self._encode_weights(text).numpy()
        return sp.csr_matrix(weights.reshape(1, -1))

    # -- catalog index (CSR matmul, RISK R3) ---------------------------------
    def build_index(self, product_ids: list[str], texts: list[str], batch_log: int = 2000):
        """Encode the whole catalog and stack it as one CSR matrix for fast search.

        Args:
            product_ids: Parallel product ids.
            texts: Parallel product texts.
            batch_log: How often to print progress (docs).
        """
        from scipy import sparse as sp

        if len(product_ids) != len(texts):
            raise ValueError("product_ids and texts must be the same length")
        rows = []
        for i, text in enumerate(texts):
            rows.append(self.encode_sparse_row(text))
            if batch_log and (i + 1) % batch_log == 0:
                print(f"[splade] encoded {i + 1}/{len(texts)} docs")
        self._doc_matrix = sp.vstack(rows, format="csr") if rows else sp.csr_matrix((0, self.vocab_size))
        self.doc_ids = list(product_ids)
        return self

    def search(self, query: str, top_k: int = 1000) -> list[tuple[str, float]]:
        """Score the query against the CSR catalog via ONE sparse matmul.

        ``scores = query_row @ doc_matrix.T`` — vectorized, not a per-doc loop.
        """
        if self._doc_matrix is None:
            raise RuntimeError("SPLADE index is empty — call build_index() first.")
        import numpy as np

        q = self.encode_sparse_row(query)              # [1, vocab] CSR
        scores = (q @ self._doc_matrix.T).toarray().ravel()  # [num_docs]
        k = min(top_k, len(self.doc_ids))
        if k == 0:
            return []
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return [(self.doc_ids[i], float(scores[i])) for i in top_idx]

    def save(self, path: str) -> None:
        """Persist the CSR doc matrix + doc_ids (npz + sidecar ids)."""
        import json

        from scipy import sparse as sp

        if self._doc_matrix is None:
            raise RuntimeError("Nothing to save — build_index() first.")
        sp.save_npz(path, self._doc_matrix)
        with open(path + ".ids.json", "w") as f:
            json.dump(self.doc_ids, f)

    def load(self, path: str) -> "SpladeEncoder":
        """Load the CSR doc matrix + doc_ids written by :meth:`save`."""
        import json

        from scipy import sparse as sp

        self._doc_matrix = sp.load_npz(path)
        with open(path + ".ids.json") as f:
            self.doc_ids = json.load(f)
        return self


def visualize_expansion(query: str, encoder: SpladeEncoder, top_k_terms: int = 20):
    """Show the terms SPLADE expands a query into (interpretability / demo, §4.3)."""
    sparse = encoder.encode(query)
    return sorted(sparse.items(), key=lambda x: x[1], reverse=True)[:top_k_terms]
