import matplotlib
matplotlib.use("Agg")  # headless-safe; must precede pyplot import

import re
import sys
import math
import json
import random
import warnings
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
MAX_TOKENS = 512
EPS = 1e-9
PLOT_PATH = "entropy_vs_density_pilot.png"
TABLE_PATH = "pilot_results_table.csv"

STRUCTURAL_KEYWORDS = [
    "due to",           # multi-word first so \b alts are unambiguous
    "contraindicated",
    "therefore",
    "inhibits",
    "increases",
    "because",
    "mechanism",
    "via",
]
_alts = "|".join(kw.replace(" ", r"\s+") for kw in STRUCTURAL_KEYWORDS)
STRUCTURAL_RE = re.compile(rf"\b(?:{_alts})\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Step 0: Hardware detection
# ---------------------------------------------------------------------------

def detect_device():
    import torch
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.bfloat16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float32   # bfloat16 is unreliable on MPS pre-macOS 14
    else:
        device = torch.device("cpu")
        dtype = torch.float32
    print(f"[device] {device}  dtype={dtype}")
    return device, dtype


# ---------------------------------------------------------------------------
# Step 1: Data ingestion — bigbio/ddi_corpus (DDI Extraction 2013)
# ---------------------------------------------------------------------------

def fetch_ddi_corpus(n: int = 100) -> Optional[list]:
    """
    Load from bigbio/ddi_corpus on HuggingFace.

    The dataset has two configs:
      - ddi_corpus_source      : original DDI-2013 schema
      - ddi_corpus_bigbio_kb   : BigBIO normalised schema

    We use ddi_corpus_source and extract passage text.
    Each document (row) contains one or more passages; we concatenate them
    into a single string so the model sees a coherent clinical paragraph.
    Documents with no passage text are skipped.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("[data] `datasets` package not installed — skipping HuggingFace fetch")
        return None

    for config in ("ddi_corpus_source", "ddi_corpus_bigbio_kb"):
        for split in ("train", "test", "validation"):
            try:
                print(f"[data] trying config={config}  split={split} …")
                ds = load_dataset(
                    "bigbio/ddi_corpus",
                    config,
                    split=split,
                )
                texts = _extract_texts_from_ddi(ds, config)
                if len(texts) >= 20:
                    random.shuffle(texts)
                    selected = texts[:n]
                    print(f"[data] loaded {len(selected)} DDI examples "
                          f"(config={config}, split={split})")
                    return selected
            except Exception as exc:
                print(f"[data]   failed ({exc.__class__.__name__}: {exc})")
                continue

    return None


def _extract_texts_from_ddi(ds, config: str) -> list:
    """Extract a flat list of non-empty text strings from a loaded DDI split."""
    texts = []

    for row in ds:
        try:
            if config == "ddi_corpus_source":
                # passages is a list of dicts: {id, type, text: [str], offsets}
                passage_texts = []
                for p in row.get("passages", []):
                    for chunk in p.get("text", []):
                        if isinstance(chunk, str) and chunk.strip():
                            passage_texts.append(chunk.strip())
                combined = " ".join(passage_texts)

            else:
                # bigbio_kb schema: passages have 'text' as a plain list
                passage_texts = []
                for p in row.get("passages", []):
                    t = p.get("text", "")
                    if isinstance(t, str) and t.strip():
                        passage_texts.append(t.strip())
                    elif isinstance(t, list):
                        passage_texts.extend(
                            s.strip() for s in t if isinstance(s, str) and s.strip()
                        )
                combined = " ".join(passage_texts)

            if len(combined.split()) >= 10:   # skip trivially short entries
                texts.append(combined)

        except Exception:
            continue

    return texts


# ---------------------------------------------------------------------------
# Fallback: synthetic DDI dataset (always works, offline-safe)
# ---------------------------------------------------------------------------

_DRUG_PAIRS = [
    ("Warfarin", "Aspirin", "CYP2C9"),
    ("Metformin", "Cimetidine", "OCT2"),
    ("Atorvastatin", "Clarithromycin", "CYP3A4"),
    ("Digoxin", "Amiodarone", "P-glycoprotein"),
    ("Ciprofloxacin", "Theophylline", "CYP1A2"),
    ("Phenytoin", "Valproate", "CYP2C9"),
    ("Metoprolol", "Verapamil", "CYP2D6"),
    ("Fluoxetine", "Tramadol", "CYP2D6"),
    ("Sildenafil", "Nitrates", "cGMP pathway"),
    ("Lithium", "Ibuprofen", "renal clearance"),
    ("Tacrolimus", "Fluconazole", "CYP3A4"),
    ("Clozapine", "Fluvoxamine", "CYP1A2"),
    ("Simvastatin", "Gemfibrozil", "CYP2C8"),
    ("Carbamazepine", "Erythromycin", "CYP3A4"),
    ("Methotrexate", "Probenecid", "OAT1/3"),
    ("Quinidine", "Digoxin", "P-glycoprotein"),
    ("Rifampin", "Oral contraceptives", "CYP3A4"),
    ("Ketoconazole", "Midazolam", "CYP3A4"),
    ("Haloperidol", "Carbamazepine", "CYP3A4"),
    ("Clopidogrel", "Omeprazole", "CYP2C19"),
]

_TEMPLATES = [
    # high structural density, relatively low entropy (formulaic)
    (
        "{d1} inhibits {enzyme} via competitive binding, which therefore increases "
        "plasma concentrations of {d2} because hepatic first-pass metabolism is "
        "reduced. This mechanism is contraindicated in patients with hepatic impairment "
        "due to accumulation risk and potential toxicity."
    ),
    # moderate density, moderate entropy
    (
        "Co-administration of {d1} and {d2} results in a clinically significant "
        "interaction mediated via {enzyme} inhibition. The mechanism involves "
        "allosteric modulation, which increases AUC of {d2} by approximately 3-fold. "
        "Therefore, dose reduction is recommended."
    ),
    # low density, higher entropy (narrative, diverse vocabulary)
    (
        "Patients receiving {d1} who are concurrently prescribed {d2} exhibit "
        "variable pharmacokinetic profiles depending on genetic polymorphisms, "
        "age, hepatic reserve, and comorbid conditions. Careful individualised "
        "assessment is advisable prior to initiation of dual therapy."
    ),
    # very low density, high entropy (advisory tone, minimal mechanism language)
    (
        "The combination of {d1} and {d2} has been evaluated in multiple randomised "
        "controlled trials with mixed outcomes. Clinical guidelines suggest monitoring "
        "and periodic reassessment rather than absolute avoidance in all populations."
    ),
    # high density, higher entropy (complex mechanistic description)
    (
        "{d1} is a potent inhibitor of {enzyme}, therefore substantially increasing "
        "systemic exposure to {d2} via impaired oxidative metabolism. Due to the "
        "narrow therapeutic index of {d2}, this interaction is contraindicated unless "
        "plasma levels are monitored. The mechanism increases risk of {d2} toxicity "
        "because clearance decreases by up to 80% in poor metabolisers."
    ),
]


def generate_synthetic(n: int = 100) -> list:
    """
    Produce n synthetic DDI strings spanning the full range of both metrics
    so the pilot is scientifically meaningful even without network access.
    """
    texts = []
    random.seed(42)
    for i in range(n):
        pair = _DRUG_PAIRS[i % len(_DRUG_PAIRS)]
        template = _TEMPLATES[i % len(_TEMPLATES)]
        texts.append(template.format(d1=pair[0], d2=pair[1], enzyme=pair[2]))
    random.shuffle(texts)
    return texts


def load_data(n: int = 100) -> tuple:
    """Returns (texts: list[str], source_label: str)."""
    print("\n[data] === Data ingestion ===")
    texts = fetch_ddi_corpus(n)
    if texts:
        return texts, "bigbio/ddi_corpus"
    print("[data] All HuggingFace attempts failed — using synthetic DDI dataset")
    synth = generate_synthetic(n)
    print(f"[data] Generated {len(synth)} synthetic DDI examples")
    print("[data] WARNING: results below are from SYNTHETIC data, not real DDI corpus.")
    return synth, "synthetic_fallback"


# ---------------------------------------------------------------------------
# Step 2: Sentence splitter setup (NLTK with regex fallback)
# ---------------------------------------------------------------------------

def setup_sent_splitter():
    try:
        import nltk
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)  # required in nltk >= 3.8.1

        def split(text: str) -> list:
            try:
                return nltk.sent_tokenize(text)
            except LookupError:
                return re.split(r"(?<=[.!?])\s+", text.strip())

        return split
    except Exception:
        return lambda text: re.split(r"(?<=[.!?])\s+", text.strip())


# ---------------------------------------------------------------------------
# Step 3: Model loading
# ---------------------------------------------------------------------------

def load_model(device, dtype):
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print(f"\n[model] Loading tokenizer for {MODEL_ID} …")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
        print("[model] Loading model weights (first run downloads ~2 GB) …")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        model = model.to(device)
        model.eval()
        print("[model] Model ready.")
        return model, tokenizer
    except Exception as exc:
        print(f"[model] Failed to load TinyLlama: {exc}")
        print("[model] Falling back to deterministic entropy surrogate.")
        return None, None


# ---------------------------------------------------------------------------
# Step 4a: Compute token entropy H(y|x)
# ---------------------------------------------------------------------------

def compute_entropy(text: str, model, tokenizer, device) -> float:
    if model is None:
        # Deterministic surrogate: mimic entropy variation via text statistics
        words = text.split()
        unique_ratio = len(set(words)) / max(len(words), 1)
        return 2.0 + unique_ratio * 3.0   # range roughly [2.0, 5.0]

    import torch

    enc = tokenizer(
        text,
        return_tensors="pt",
        max_length=MAX_TOKENS,
        truncation=True,
        padding=False,
    )
    input_ids = enc["input_ids"].to(device)

    if input_ids.shape[1] < 2:
        return 0.0

    with torch.no_grad():
        logits = model(input_ids).logits[0].float()  # [seq_len, vocab]

    probs = torch.softmax(logits, dim=-1)
    token_entropy = -torch.sum(probs * torch.log(probs + EPS), dim=-1)
    return token_entropy.mean().item()


# ---------------------------------------------------------------------------
# Step 4b: Compute structural density
# ---------------------------------------------------------------------------

def compute_structural_density(text: str, sent_split) -> float:
    sentences = [s.strip() for s in sent_split(text) if s.strip()]
    n_sentences = max(len(sentences), 1)
    hits = len(STRUCTURAL_RE.findall(text))
    return hits / n_sentences


# ---------------------------------------------------------------------------
# Step 5: Analysis
# ---------------------------------------------------------------------------

def run_analysis(entropies: list, densities: list) -> dict:
    from scipy.stats import pearsonr, spearmanr

    e = np.array(entropies)
    d = np.array(densities)

    if np.std(e) < 1e-10 or np.std(d) < 1e-10:
        print("[warn] Near-zero variance in one metric — correlation unreliable.")
        r, p_r, rho, p_rho = float("nan"), 1.0, float("nan"), 1.0
    else:
        r, p_r = pearsonr(e, d)
        rho, p_rho = spearmanr(e, d)

    med_e, med_d = float(np.median(e)), float(np.median(d))

    q = {"Q1": 0, "Q2": 0, "Q3": 0, "Q4": 0}
    for ei, di in zip(e, d):
        if ei >= med_e and di >= med_d:
            q["Q1"] += 1   # high entropy, high density  (top-right)
        elif ei < med_e and di >= med_d:
            q["Q2"] += 1   # low entropy,  high density  (top-left)
        elif ei < med_e and di < med_d:
            q["Q3"] += 1   # low entropy,  low density   (bottom-left)
        else:
            q["Q4"] += 1   # high entropy, low density   (bottom-right)

    n = len(entropies)
    q_pct = {k: 100 * v / n for k, v in q.items()}

    if math.isnan(r):
        decision = "DECISION: AMBIGUOUS — variance too low for reliable correlation"
    elif r > 0.8:
        decision = "DECISION: DROP THESIS"
    elif r < 0.6:
        decision = "DECISION: PROCEED WITH JOINT METRIC"
    else:
        decision = "DECISION: AMBIGUOUS — collect more data"

    return dict(
        pearson_r=float(r), pearson_p=float(p_r),
        spearman_rho=float(rho), spearman_p=float(p_rho),
        median_entropy=med_e, median_density=med_d,
        quadrant_pct=q_pct,
        decision=decision,
    )


# ---------------------------------------------------------------------------
# Step 6: Scatter plot
# ---------------------------------------------------------------------------

def make_scatter_plot(entropies: list, densities: list, results: dict) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        entropies, densities,
        alpha=0.65, edgecolors="steelblue", facecolors="lightblue", s=55,
    )

    med_e = results["median_entropy"]
    med_d = results["median_density"]

    ax.axvline(med_e, color="red", linestyle="--", linewidth=1.2,
               label=f"median entropy = {med_e:.3f}")
    ax.axhline(med_d, color="darkgreen", linestyle="--", linewidth=1.2,
               label=f"median density = {med_d:.3f}")

    # Quadrant labels — position after axes limits are set by scatter()
    xl, xr = ax.get_xlim()
    yb, yt = ax.get_ylim()
    offset_x = (xr - xl) * 0.04
    offset_y = (yt - yb) * 0.04
    q = results["quadrant_pct"]

    ax.text(med_e + offset_x, yt - offset_y * 4,
            f"Q1\n{q['Q1']:.1f}%", va="top", color="dimgray", fontsize=9)
    ax.text(xl + offset_x, yt - offset_y * 4,
            f"Q2\n{q['Q2']:.1f}%", va="top", color="dimgray", fontsize=9)
    ax.text(xl + offset_x, yb + offset_y,
            f"Q3\n{q['Q3']:.1f}%", va="bottom", color="dimgray", fontsize=9)
    ax.text(med_e + offset_x, yb + offset_y,
            f"Q4\n{q['Q4']:.1f}%", va="bottom", color="dimgray", fontsize=9)

    r = results["pearson_r"]
    rho = results["spearman_rho"]
    ax.set_title(
        f"Token Entropy vs Structural Density — DDI Corpus pilot (n={len(entropies)})\n"
        f"Pearson r={r:.3f}   Spearman ρ={rho:.3f}",
        fontsize=11,
    )
    ax.set_xlabel("Mean Token Entropy  H(y|x)", fontsize=11)
    ax.set_ylabel("Structural Density  (keyword hits / sentence)", fontsize=11)
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)
    print(f"[plot] saved → {PLOT_PATH}")


# ---------------------------------------------------------------------------
# Step 7a: Per-example table (CSV + console preview)
# ---------------------------------------------------------------------------

def save_results_table(
    texts: list,
    entropies: list,
    densities: list,
    results: dict,
    data_source: str,
) -> None:
    import csv

    med_e = results["median_entropy"]
    med_d = results["median_density"]

    def quadrant(e, d):
        if e >= med_e and d >= med_d:
            return "Q1"
        elif e < med_e and d >= med_d:
            return "Q2"
        elif e < med_e and d < med_d:
            return "Q3"
        return "Q4"

    rows = []
    for i, (text, e, d) in enumerate(zip(texts, entropies, densities)):
        rows.append({
            "idx": i + 1,
            "data_source": data_source,
            "text_snippet": text[:100].replace("\n", " "),
            "entropy": round(e, 4),
            "structural_density": round(d, 4),
            "quadrant": quadrant(e, d),
        })

    with open(TABLE_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["idx", "data_source", "text_snippet",
                           "entropy", "structural_density", "quadrant"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[table] saved → {TABLE_PATH}  ({len(rows)} rows)")

    # Console preview: top 10 rows sorted by entropy descending
    print(f"\n{'idx':>4}  {'entropy':>8}  {'density':>8}  {'Q':>2}  text snippet")
    print("-" * 75)
    for r in sorted(rows, key=lambda x: x["entropy"], reverse=True)[:10]:
        print(f"{r['idx']:>4}  {r['entropy']:>8.4f}  {r['structural_density']:>8.4f}"
              f"  {r['quadrant']:>2}  {r['text_snippet'][:48]}…")
    print(f"  … (full {len(rows)}-row table in {TABLE_PATH})")


# ---------------------------------------------------------------------------
# Step 7b: Aggregate console output
# ---------------------------------------------------------------------------

def print_results(results: dict, data_source: str = "") -> None:
    sep = "=" * 58
    print(f"\n{sep}")
    print("  PILOT RESULTS: Entropy vs Structural Density (DDI)")
    if data_source:
        print(f"  Data source: {data_source}")
    print(sep)
    print(f"  Pearson  r   = {results['pearson_r']:.4f}  "
          f"(p = {results['pearson_p']:.3g})")
    print(f"  Spearman ρ   = {results['spearman_rho']:.4f}  "
          f"(p = {results['spearman_p']:.3g})")
    print()
    print("  Quadrant distribution (median split):")
    for q, pct in results["quadrant_pct"].items():
        label = {
            "Q1": "high entropy | high density  (top-right)",
            "Q2": "low entropy  | high density  (top-left) ",
            "Q3": "low entropy  | low density   (bot-left) ",
            "Q4": "high entropy | low density   (bot-right)",
        }[q]
        print(f"    {q}  [{label}]  {pct:.1f}%")
    print()
    print(f"  {results['decision']}")
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # 1. Data
    raw_texts, data_source = load_data(n=100)

    # 2. Infrastructure
    sent_split = setup_sent_splitter()
    device, dtype = detect_device()
    model, tokenizer = load_model(device, dtype)

    # 3. Metric computation
    from tqdm import tqdm
    texts, entropies, densities = [], [], []

    print(f"\n[metrics] Computing entropy + structural density for {len(raw_texts)} examples …")
    for text in tqdm(raw_texts, desc="examples", unit="ex"):
        if not isinstance(text, str) or not text.strip():
            continue
        texts.append(text)
        entropies.append(compute_entropy(text, model, tokenizer, device))
        densities.append(compute_structural_density(text, sent_split))

    if len(entropies) < 10:
        sys.exit("[error] Fewer than 10 valid examples — cannot proceed with analysis.")

    # 4. Analysis + output
    results = run_analysis(entropies, densities)
    make_scatter_plot(entropies, densities, results)
    save_results_table(texts, entropies, densities, results, data_source)
    print_results(results, data_source)


if __name__ == "__main__":
    main()
