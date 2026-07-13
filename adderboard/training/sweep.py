"""
Multi-seed sweep for grokking detection. Runs short training with many seeds,
identifies promising seeds (loss dropping faster), runs them longer.

Strategy:
  1. Phase 1: 20 seeds × 2000 steps → pick top-5 by loss improvement
  2. Phase 2: top-5 seeds × 20000 steps → pick best
  3. Phase 3: best seed × full run (100K+ steps)
"""

import subprocess
import sys
import json
import os
import math
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = BASE_DIR / "retrain" / "weights"
TRAIN_SCRIPT = BASE_DIR / "retrain" / "retrain.py"

PHASE1_SEEDS = list(range(10))
PHASE1_STEPS = 2000
PHASE2_STEPS = 20000
PHASE3_STEPS = 200000
TOP_K = 5
BATCH = 128
LR = 0.01

def run_training(seed, steps, output_dir, extra_args=None):
    """Run retrain.py and return (seed, final_loss, best_acc, steps_run)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--seed", str(seed),
        "--epochs", str(steps),
        "--batch-size", str(BATCH),
        "--lr", str(LR),
        "--eval-interval", str(steps),  # eval only at end
        "--eval-samples", "200",
        "--output", str(output_dir),
    ]
    if extra_args:
        cmd.extend(extra_args)
    
    print(f"[seed={seed}] Running {steps} steps...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=steps * 0.02 + 60)
    
    # Parse output for loss and accuracy
    lines = result.stdout.split("\n") + result.stderr.split("\n")
    losses = []
    best_acc = 0.0
    for line in lines:
        if "loss" in line and "step" in line and "| loss" in line:
            try:
                parts = line.split("|")
                loss_str = parts[1].strip().split()[1]
                losses.append(float(loss_str))
            except:
                pass
        if "FINAL:" in line and "exact=" in line:
            try:
                acc_str = line.split("exact=")[1].split()[0]
                best_acc = float(acc_str)
            except:
                pass
    
    final_loss = losses[-1] if losses else 99.0
    return {
        "seed": seed,
        "steps": steps,
        "final_loss": final_loss,
        "best_acc": best_acc,
        "n_losses": len(losses),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--base-seed", type=int, default=None,
                        help="Skip to phase 3 with a specific seed")
    parser.add_argument("--n-seeds", type=int, default=20,
                        help="Number of seeds for phase 1")
    args = parser.parse_args()
    
    results_file = WEIGHTS_DIR / "sweep_results.json"
    
    if args.base_seed:
        print(f"=== Phase 3: Long run for seed {args.base_seed} ===")
        result = run_training(args.base_seed, PHASE3_STEPS, 
                             WEIGHTS_DIR / f"sweep_seed{args.base_seed}_long")
        print(json.dumps(result, indent=2))
        return
    
    if args.phase == 1:
        print(f"=== Phase 1: Sweep {args.n_seeds} seeds × {PHASE1_STEPS} steps ===")
        print(f"Seeds: {PHASE1_SEEDS[:args.n_seeds]}")
        print()
        
        results = []
        for seed in PHASE1_SEEDS[:args.n_seeds]:
            r = run_training(seed, PHASE1_STEPS, 
                            WEIGHTS_DIR / f"sweep_seed{seed}_phase1")
            results.append(r)
            print(f"  seed={seed:2d}: loss={r['final_loss']:.4f}, acc={r['best_acc']:.4f}")
        
        # Sort by loss (lower is better)
        results.sort(key=lambda r: r["final_loss"])
        
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nTop {TOP_K} seeds:")
        for r in results[:TOP_K]:
            print(f"  seed={r['seed']:2d}: loss={r['final_loss']:.4f}")
        
        print(f"\nSaved to {results_file}")
        print(f"Run: python retrain/sweep.py --phase 2")
    
    elif args.phase == 2:
        with open(results_file) as f:
            results = json.load(f)
        
        top_seeds = [r["seed"] for r in sorted(results, key=lambda r: r["final_loss"])[:TOP_K]]
        
        print(f"=== Phase 2: Top {TOP_K} seeds × {PHASE2_STEPS} steps ===")
        print(f"Seeds: {top_seeds}")
        print()
        
        phase2_results = []
        for seed in top_seeds:
            r = run_training(seed, PHASE2_STEPS, 
                            WEIGHTS_DIR / f"sweep_seed{seed}_phase2")
            phase2_results.append(r)
            print(f"  seed={seed:2d}: loss={r['final_loss']:.4f}, acc={r['best_acc']:.4f}")
        
        phase2_results.sort(key=lambda r: r["best_acc"], reverse=True)
        
        with open(WEIGHTS_DIR / "sweep_phase2_results.json", "w") as f:
            json.dump(phase2_results, f, indent=2)
        
        print(f"\nBest seed: {phase2_results[0]['seed']}, acc={phase2_results[0]['best_acc']:.4f}")
        print(f"Run: python retrain/sweep.py --base-seed {phase2_results[0]['seed']}")
    
    elif args.phase == 3:
        # Find best from phase 2
        phase2_file = WEIGHTS_DIR / "sweep_phase2_results.json"
        with open(phase2_file) as f:
            p2_results = json.load(f)
        
        best = sorted(p2_results, key=lambda r: r["best_acc"], reverse=True)[0]
        seed = best["seed"]
        
        print(f"=== Phase 3: Long run for best seed {seed} ===")
        r = run_training(seed, PHASE3_STEPS,
                        WEIGHTS_DIR / f"sweep_seed{seed}_long",
                        extra_args=["--eval-interval", "10000"])
        print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
