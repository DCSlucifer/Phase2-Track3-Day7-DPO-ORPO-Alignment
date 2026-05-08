from __future__ import annotations
from pathlib import Path
from typing import Optional
import typer
from rich import print
from rich.table import Table
from .config import load_config
from .data import load_jsonl, load_jsonl_with_diagnostics, split_by_prompt
from .evaluate import pairwise_accuracy, reward_margin, reward_std, score_examples, write_metrics
from .trainers import TrainingConfig, PreferenceTrainer

app = typer.Typer(help="Preference alignment lab CLI")


@app.command()
def validate(data: Path) -> None:
    """Validate a preference JSONL dataset with diagnostics."""
    result = load_jsonl_with_diagnostics(data)

    print(f"\n[bold]Dataset Validation Report[/bold]")
    print(f"  File: {data}")
    print(f"  Total lines: {result.total_lines}")
    print(f"  Valid examples: [green]{len(result.examples)}[/green]")

    if result.errors:
        print(f"\n  [red]Errors ({len(result.errors)}):[/red]")
        for e in result.errors:
            print(f"    ✗ {e}")

    if result.warnings:
        print(f"\n  [yellow]Warnings ({len(result.warnings)}):[/yellow]")
        for w in result.warnings:
            print(f"    ⚠ {w}")

    if result.pii_warnings:
        print(f"\n  [magenta]PII Warnings ({len(result.pii_warnings)}):[/magenta]")
        for p in result.pii_warnings:
            print(f"    🔒 {p}")

    if result.duplicate_prompts:
        print(f"\n  [yellow]Duplicate prompts: {len(result.duplicate_prompts)}[/yellow]")

    if not result.errors and not result.warnings:
        print(f"\n  [green]✓ All checks passed![/green]")


@app.command()
def evaluate(
    config: Path,
    scorer: str = typer.Option("combined", help="Scorer: mock | length | keyword | combined"),
) -> None:
    """Run evaluation pipeline and save metrics."""
    cfg = load_config(config)
    examples = load_jsonl(cfg["paths"]["train_data"])

    # Score examples
    chosen_scores, rejected_scores = score_examples(examples, scorer=scorer)

    # Compute metrics
    accuracy_metrics = pairwise_accuracy(examples, chosen_scores, rejected_scores)
    margin = reward_margin(chosen_scores, rejected_scores)
    chosen_std = reward_std(chosen_scores)
    rejected_std = reward_std(rejected_scores)

    metrics = {
        "scorer": scorer,
        "dataset_size": len(examples),
        **accuracy_metrics,
        "reward_margin": round(margin, 4),
        "chosen_score_std": round(chosen_std, 4),
        "rejected_score_std": round(rejected_std, 4),
    }

    out = write_metrics(metrics, cfg["paths"]["output_dir"])

    # Pretty print
    table = Table(title="Evaluation Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    for k, v in metrics.items():
        table.add_row(k, str(v))
    print(table)
    print(f"\n[green]Wrote metrics to {out}[/green]")


@app.command()
def train(
    config: Path,
    method: str = typer.Option("both", help="Training method: dpo | orpo | both"),
    steps: int = typer.Option(50, help="Number of training steps"),
) -> None:
    """Run mock training and save results."""
    cfg = load_config(config)
    training_cfg = cfg.get("training", {})

    tc = TrainingConfig(
        method=method,
        beta=training_cfg.get("beta", 0.1),
        lambda_orpo=training_cfg.get("lambda_orpo", 0.1),
        max_length=training_cfg.get("max_length", 512),
        batch_size=training_cfg.get("batch_size", 2),
        num_steps=steps,
        output_dir=cfg["paths"]["output_dir"],
    )

    trainer = PreferenceTrainer(tc)
    results = trainer.train()

    for r in results:
        table = Table(title=f"{r.method.upper()} Training Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Method", r.method)
        table.add_row("Steps", str(r.steps))
        table.add_row("Initial Loss", f"{r.loss_history[0]:.6f}")
        table.add_row("Final Loss", f"{r.final_loss:.6f}")
        table.add_row("Loss Reduction", f"{r.loss_history[0] - r.final_loss:.6f}")
        table.add_row("Time", f"{r.elapsed_seconds:.3f}s")
        print(table)

    print(f"\n[green]Training results saved to {cfg['paths']['output_dir']}[/green]")


@app.command()
def split(
    data: Path,
    ratio: float = typer.Option(0.2, help="Validation ratio"),
    seed: int = typer.Option(42, help="Random seed"),
) -> None:
    """Show train/val split info for a dataset."""
    examples = load_jsonl(data)
    train_set, val_set = split_by_prompt(examples, validation_ratio=ratio, seed=seed)

    train_prompts = {ex.prompt for ex in train_set}
    val_prompts = {ex.prompt for ex in val_set}
    overlap = train_prompts & val_prompts

    print(f"\n[bold]Split Results[/bold] (seed={seed}, ratio={ratio})")
    print(f"  Total examples: {len(examples)}")
    print(f"  Train: [green]{len(train_set)}[/green] ({len(train_prompts)} unique prompts)")
    print(f"  Val:   [blue]{len(val_set)}[/blue] ({len(val_prompts)} unique prompts)")
    print(f"  Prompt overlap: [{'red' if overlap else 'green'}]{len(overlap)}[/{'red' if overlap else 'green'}]")

    if overlap:
        print(f"  [red]⚠ Data leakage detected![/red]")
    else:
        print(f"  [green]✓ No data leakage — prompts are disjoint[/green]")


if __name__ == "__main__":
    app()
