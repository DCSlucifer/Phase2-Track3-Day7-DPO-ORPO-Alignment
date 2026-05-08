"""FastAPI backend for the Preference Alignment Lab Dashboard."""
from __future__ import annotations
import sys
import os
from pathlib import Path

# Add project root to path so we can import preference_lab
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.chdir(str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import numpy as np

from preference_lab.data import load_jsonl, load_jsonl_with_diagnostics, split_by_prompt
from preference_lab.losses import dpo_loss, orpo_loss
from preference_lab.evaluate import (
    pairwise_accuracy, reward_margin, reward_std,
    score_examples, write_metrics,
)
from preference_lab.trainers import TrainingConfig, PreferenceTrainer
from preference_lab.safety import run_safety_regression, save_safety_report
from preference_lab.comparison import run_comparison, save_comparison
from preference_lab.config import load_config

app = FastAPI(title="Preference Alignment Lab", version="1.0.0")

# Serve static files
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/dataset")
def get_dataset():
    """Load and validate the dataset with full diagnostics."""
    result = load_jsonl_with_diagnostics("data/sample_preferences.jsonl")
    examples = [
        {
            "prompt": ex.prompt,
            "chosen": ex.chosen,
            "rejected": ex.rejected,
            "metadata": ex.metadata,
        }
        for ex in result.examples
    ]
    return {
        "examples": examples,
        "total_lines": result.total_lines,
        "valid_count": len(result.examples),
        "errors": result.errors,
        "warnings": result.warnings,
        "pii_warnings": result.pii_warnings,
        "duplicate_prompts": result.duplicate_prompts,
    }


@app.get("/api/dataset/split")
def get_split(ratio: float = 0.2, seed: int = 42):
    """Show train/val split with prompt grouping info."""
    examples = load_jsonl("data/sample_preferences.jsonl")
    train, val = split_by_prompt(examples, validation_ratio=ratio, seed=seed)

    train_prompts = list({ex.prompt for ex in train})
    val_prompts = list({ex.prompt for ex in val})
    overlap = set(train_prompts) & set(val_prompts)

    return {
        "total": len(examples),
        "train_count": len(train),
        "val_count": len(val),
        "train_prompts": len(train_prompts),
        "val_prompts": len(val_prompts),
        "overlap": len(overlap),
        "has_leakage": len(overlap) > 0,
        "train_examples": [{"prompt": e.prompt[:80], "chosen": e.chosen[:80], "rejected": e.rejected[:80]} for e in train[:5]],
        "val_examples": [{"prompt": e.prompt[:80], "chosen": e.chosen[:80], "rejected": e.rejected[:80]} for e in val[:5]],
    }


@app.get("/api/loss/dpo")
def compute_dpo(
    policy_chosen: float = -0.5,
    policy_rejected: float = -1.5,
    ref_chosen: float = -0.6,
    ref_rejected: float = -1.0,
    beta: float = 0.1,
):
    """Compute DPO loss with given parameters."""
    loss = dpo_loss(
        np.array([policy_chosen]),
        np.array([policy_rejected]),
        np.array([ref_chosen]),
        np.array([ref_rejected]),
        beta=beta,
    )
    policy_log_ratio = policy_chosen - policy_rejected
    ref_log_ratio = ref_chosen - ref_rejected
    logits = beta * (policy_log_ratio - ref_log_ratio)

    return {
        "loss": round(loss, 6),
        "policy_log_ratio": round(policy_log_ratio, 6),
        "ref_log_ratio": round(ref_log_ratio, 6),
        "logits": round(logits, 6),
        "beta": beta,
    }


@app.get("/api/loss/orpo")
def compute_orpo(
    sft_nll: float = 1.0,
    chosen_logps: float = -0.5,
    rejected_logps: float = -1.5,
    lambda_orpo: float = 0.1,
):
    """Compute ORPO loss with given parameters."""
    loss = orpo_loss(
        np.array([sft_nll]),
        np.array([chosen_logps]),
        np.array([rejected_logps]),
        lambda_orpo=lambda_orpo,
    )
    return {
        "loss": round(loss, 6),
        "sft_nll": sft_nll,
        "chosen_logps": chosen_logps,
        "rejected_logps": rejected_logps,
        "lambda_orpo": lambda_orpo,
    }


@app.get("/api/train")
def run_training(
    method: str = "both",
    steps: int = 50,
    beta: float = 0.1,
    lambda_orpo: float = 0.1,
    batch_size: int = 4,
):
    """Run mock training and return loss curves."""
    tc = TrainingConfig(
        method=method,
        beta=beta,
        lambda_orpo=lambda_orpo,
        batch_size=batch_size,
        num_steps=steps,
        output_dir="outputs",
    )
    trainer = PreferenceTrainer(tc)
    results = trainer.train()

    return {
        "results": [
            {
                "method": r.method,
                "loss_history": [round(l, 6) for l in r.loss_history],
                "final_loss": round(r.final_loss, 6),
                "steps": r.steps,
                "elapsed_seconds": round(r.elapsed_seconds, 4),
                "initial_loss": round(r.loss_history[0], 6),
                "loss_reduction": round(r.loss_history[0] - r.final_loss, 6),
            }
            for r in results
        ]
    }


@app.get("/api/evaluate")
def run_evaluation(scorer: str = "combined"):
    """Run full evaluation pipeline."""
    examples = load_jsonl("data/sample_preferences.jsonl")
    chosen_scores, rejected_scores = score_examples(examples, scorer=scorer)

    acc = pairwise_accuracy(examples, chosen_scores, rejected_scores)
    margin = reward_margin(chosen_scores, rejected_scores)
    c_std = reward_std(chosen_scores)
    r_std = reward_std(rejected_scores)

    # Per-example breakdown
    per_example = []
    for i, ex in enumerate(examples):
        per_example.append({
            "prompt": ex.prompt[:80],
            "chosen_score": round(chosen_scores[i], 4),
            "rejected_score": round(rejected_scores[i], 4),
            "margin": round(chosen_scores[i] - rejected_scores[i], 4),
            "correct": chosen_scores[i] > rejected_scores[i],
        })

    metrics = {
        "scorer": scorer,
        "dataset_size": len(examples),
        **{k: round(v, 4) if isinstance(v, float) else v for k, v in acc.items()},
        "reward_margin": round(margin, 4),
        "chosen_score_std": round(c_std, 4),
        "rejected_score_std": round(r_std, 4),
    }

    write_metrics(metrics, "outputs")

    return {
        "metrics": metrics,
        "per_example": per_example,
    }


@app.get("/api/compare")
def run_dpo_vs_orpo(
    beta: float = 0.1,
    lambda_orpo: float = 0.1,
    steps: int = 50,
    batch_size: int = 4,
):
    """Run DPO vs ORPO comparison."""
    comparison = run_comparison(
        beta=beta,
        lambda_orpo=lambda_orpo,
        num_steps=steps,
        batch_size=batch_size,
    )
    save_comparison(comparison, "outputs")

    return {
        "winner": comparison.winner,
        "analysis": comparison.analysis,
        "dpo": {
            "loss_history": [round(l, 6) for l in comparison.dpo.loss_history] if comparison.dpo else [],
            "final_loss": round(comparison.dpo.final_loss, 6) if comparison.dpo else 0,
        },
        "orpo": {
            "loss_history": [round(l, 6) for l in comparison.orpo.loss_history] if comparison.orpo else [],
            "final_loss": round(comparison.orpo.final_loss, 6) if comparison.orpo else 0,
        },
    }


@app.get("/api/safety")
def run_safety(mode: str = "both"):
    """Run safety regression tests."""
    results = {}

    if mode in ("before", "both"):
        before = run_safety_regression(safe_mode=False)
        results["before"] = {
            "overall_score": round(before.overall_score, 4),
            "passed": before.passed,
            "passed_prompts": before.passed_prompts,
            "total_prompts": before.total_prompts,
            "results": [
                {
                    "category": s.category,
                    "prompt": s.prompt,
                    "response": s.response,
                    "safety_score": round(s.safety_score, 4),
                    "passed": s.passed,
                    "safety_hits": s.safety_keyword_hits,
                    "danger_hits": s.danger_keyword_hits,
                }
                for s in before.scores
            ],
        }

    if mode in ("after", "both"):
        after = run_safety_regression(safe_mode=True)
        save_safety_report(after, "outputs")
        results["after"] = {
            "overall_score": round(after.overall_score, 4),
            "passed": after.passed,
            "passed_prompts": after.passed_prompts,
            "total_prompts": after.total_prompts,
            "results": [
                {
                    "category": s.category,
                    "prompt": s.prompt,
                    "response": s.response,
                    "safety_score": round(s.safety_score, 4),
                    "passed": s.passed,
                    "safety_hits": s.safety_keyword_hits,
                    "danger_hits": s.danger_keyword_hits,
                }
                for s in after.scores
            ],
        }

    return results


@app.get("/api/config")
def get_config():
    """Return current YAML config."""
    cfg = load_config("configs/local.yaml")
    return cfg


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765)
