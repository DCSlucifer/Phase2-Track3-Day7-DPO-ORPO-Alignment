"""Collect all metrics for report documentation."""
import sys
sys.stdout.reconfigure(encoding="utf-8")

from preference_lab.data import load_jsonl_with_diagnostics, split_by_prompt
from preference_lab.evaluate import pairwise_accuracy, reward_margin, reward_std, score_examples
from preference_lab.safety import run_safety_regression
from preference_lab.comparison import run_comparison
from preference_lab.trainers import TrainingConfig, PreferenceTrainer

# 1. Dataset validation
r = load_jsonl_with_diagnostics("data/sample_preferences.jsonl")
print("=== DATASET ===")
print("Total lines:", r.total_lines)
print("Valid examples:", len(r.examples))
print("Errors:", len(r.errors))
print("Warnings:", len(r.warnings))
print("PII warnings:", len(r.pii_warnings))
print("Duplicates:", len(r.duplicate_prompts))

# Domain stats
domains = {}
for ex in r.examples:
    d = ex.metadata.get("domain", "unknown")
    domains[d] = domains.get(d, 0) + 1
print("Domains:", domains)

# 2. Split
train, val = split_by_prompt(r.examples, validation_ratio=0.2, seed=42)
train_p = set()
val_p = set()
for e in train:
    train_p.add(e.prompt.strip().lower())
for e in val:
    val_p.add(e.prompt.strip().lower())
overlap = train_p & val_p
print()
print("=== SPLIT ===")
print("Train:", len(train), "examples,", len(train_p), "unique prompts")
print("Val:", len(val), "examples,", len(val_p), "unique prompts")
print("Overlap:", len(overlap))

# 3. Evaluation
print()
print("=== EVALUATION ===")
for scorer in ["combined", "length", "keyword", "mock"]:
    cs, rs = score_examples(r.examples, scorer=scorer)
    acc = pairwise_accuracy(r.examples, cs, rs)
    margin = reward_margin(cs, rs)
    std_c = reward_std(cs)
    std_r = reward_std(rs)
    a = acc["accuracy"] * 100
    w = acc["win_count"]
    lo = acc["loss_count"]
    t = acc["tie_count"]
    print(scorer + ":", "acc=" + str(round(a, 1)) + "%",
          "wins=" + str(w), "losses=" + str(lo), "ties=" + str(t),
          "margin=" + str(round(margin, 4)),
          "chosen_std=" + str(round(std_c, 4)),
          "rejected_std=" + str(round(std_r, 4)))

# First 3 examples with combined scorer
cs, rs = score_examples(r.examples, scorer="combined")
print()
print("=== SAMPLE SCORES (combined) ===")
for i in range(min(5, len(r.examples))):
    ex = r.examples[i]
    print("  [" + str(i + 1) + "] prompt=" + repr(ex.prompt[:60]))
    print("      chosen=" + str(round(cs[i], 4)) + " rejected=" + str(round(rs[i], 4))
          + " margin=" + str(round(cs[i] - rs[i], 4))
          + " correct=" + str(cs[i] > rs[i]))

# 4. Training
print()
print("=== TRAINING ===")
tc = TrainingConfig(method="both", beta=0.1, lambda_orpo=0.1, num_steps=50, batch_size=4, output_dir="outputs")
trainer = PreferenceTrainer(tc)
results = trainer.train()
for res in results:
    init = res.loss_history[0]
    fin = res.final_loss
    red = init - fin
    print(res.method + ": initial=" + str(round(init, 6))
          + " final=" + str(round(fin, 6))
          + " reduction=" + str(round(red, 6))
          + " steps=" + str(res.steps)
          + " time=" + str(round(res.elapsed_seconds, 4)) + "s")

# 5. Safety
print()
print("=== SAFETY ===")
before = run_safety_regression(safe_mode=False)
after = run_safety_regression(safe_mode=True)
print("Before: score=" + str(round(before.overall_score, 4))
      + " passed=" + str(before.passed_prompts) + "/" + str(before.total_prompts))
print("After: score=" + str(round(after.overall_score, 4))
      + " passed=" + str(after.passed_prompts) + "/" + str(after.total_prompts))
for s in before.scores:
    print("  [BEFORE] " + s.category + ": score=" + str(round(s.safety_score, 4))
          + " passed=" + str(s.passed)
          + " safety_hits=" + str(s.safety_keyword_hits)
          + " danger_hits=" + str(s.danger_keyword_hits))
for s in after.scores:
    print("  [AFTER] " + s.category + ": score=" + str(round(s.safety_score, 4))
          + " passed=" + str(s.passed)
          + " safety_hits=" + str(s.safety_keyword_hits)
          + " danger_hits=" + str(s.danger_keyword_hits))

# 6. Comparison
print()
print("=== COMPARISON ===")
cmp = run_comparison(beta=0.1, lambda_orpo=0.1, num_steps=50, batch_size=4)
print("Winner:", cmp.winner)
print("DPO: initial=" + str(round(cmp.dpo.loss_history[0], 6))
      + " final=" + str(round(cmp.dpo.final_loss, 6)))
print("ORPO: initial=" + str(round(cmp.orpo.loss_history[0], 6))
      + " final=" + str(round(cmp.orpo.final_loss, 6)))
print("Analysis:", cmp.analysis)
