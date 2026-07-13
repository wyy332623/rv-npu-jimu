"""
FP32 training of a d=4, 1h transformer for 10-digit addition.
Quantize trained weights to FP16 and verify.

Based on dimopep_140p — a 140-param model that achieved 100% accuracy
while keeping all weights <100 (FP16-safe). Uses RoPE, SwiGLU, RMSNorm,
tied K=V, tied O=Q^T, and tied LM head.

Training technique from tbukic's M10S submission:
  1. Train with cosine LR schedule (AdamW, lr=0.01 → lr/10)
  2. Targeted fine-tuning on error pairs
  3. Multiple seeds, keep best
  4. All weights naturally <100 → FP16 quantization is lossless

Constraints:
  - d_model=4, head_dim=4, 1 head, 1 KV head
  - FF dim=4, SwiGLU activation
  - RoPE theta=3.0
  - RMSNorm (not LayerNorm)
  - Tied K=V, O=Q^T, LM head = embed.T
  - Sinusoidal PE (not learned)
  - LSB-first reversed digit format

Usage:
  # Train from scratch
  python retrain/retrain.py --epochs 100000 --batch-size 128 --lr 0.01 --seed 42

  # Resume from checkpoint
  python retrain/retrain.py --resume retrain/weights/retrained_best.pt --epochs 20000 --lr 0.0003

  # Targeted fine-tuning
  python retrain/retrain.py --resume retrain/weights/retrained_best.pt --targeted --lr 0.0003 --targeted-steps 5000
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import argparse
import os
import sys
import time
import csv
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adderboard.layout.layout_130p import (
    PROMPT_LEN, OUTPUT_DIGITS, VOCAB_SIZE, MODEL_DIM,
    N_HEADS, HEAD_DIM, FF_DIM, dram_addr, 
    encode_prompt as _encode_prompt,
    decode_output as _decode_output,
)

# ── Model ────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


def apply_rope(x, theta=3.0):
    """Apply rotary positional encoding to the last dimension."""
    B, T, H, head_dim = x.shape
    
    positions = torch.arange(T, device=x.device, dtype=torch.float32)
    
    # RoPE theta: frequency for each dimension pair
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim // 2, device=x.device, dtype=torch.float32) / (head_dim // 2)))
    angles = positions.unsqueeze(1) * freqs.unsqueeze(0)  # [T, hd/2]
    
    cos_vals = torch.cos(angles).unsqueeze(0).unsqueeze(2)  # [1, T, 1, hd/2]
    sin_vals = torch.sin(angles).unsqueeze(0).unsqueeze(2)
    
    # Reshape into paired dimensions: [B, T, H, hd/2, 2]
    x_pairs = x.reshape(B, T, H, head_dim // 2, 2)
    x0, x1 = x_pairs[..., 0], x_pairs[..., 1]
    
    out0 = x0 * cos_vals - x1 * sin_vals
    out1 = x0 * sin_vals + x1 * cos_vals
    
    return torch.stack([out0, out1], dim=-1).reshape(B, T, H, head_dim)


class TinyAdderQwen3(nn.Module):
    """1-layer Qwen3-style decoder: d=4, 1h, hd=4, ff=4, RoPE, SwiGLU, RMSNorm.
    
    Tied: K=V, O=Q^T, LM head = embed.T
    """
    def __init__(self, d_model=4, head_dim=4, ff_dim=4, rope_theta=3.0, vocab_size=10):
        super().__init__()
        self.d_model = d_model
        self.head_dim = head_dim
        self.vocab_size = vocab_size
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Norms
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.norm_final = RMSNorm(d_model)
        
        # QKV: Q projection + KV projection (tied K=V)
        self.W_q = nn.Linear(d_model, head_dim, bias=False)
        self.W_kv = nn.Linear(d_model, head_dim, bias=False)
        
        # QK norms
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
        
        # FFN: SwiGLU (gate, up, down)
        self.W_gate = nn.Linear(d_model, ff_dim, bias=False)
        self.W_up = nn.Linear(d_model, ff_dim, bias=False)
        self.W_down = nn.Linear(ff_dim, d_model, bias=False)
        
        self.rope_theta = rope_theta
    
    def forward(self, token_ids):
        """token_ids: [B, T] → logits: [B, T, V]"""
        B, T = token_ids.shape
        
        # Embed (no sinusoidal PE — RoPE provides positional info)
        x = self.embedding(token_ids)
        
        # ── Attention ──
        h = self.norm1(x)  # [B, T, d_model]
        
        q = self.W_q(h).reshape(B, T, 1, self.head_dim)   # [B, T, 1, hd]
        kv = self.W_kv(h).reshape(B, T, 1, self.head_dim)
        k = v = kv
        
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = apply_rope(q, self.rope_theta)
        k = apply_rope(k, self.rope_theta)
        
        q = q.transpose(1, 2)  # [B, 1, T, hd]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale  # [B, 1, T, T]
        
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(attn, dim=-1)
        
        attn_out = (attn @ v).transpose(1, 2).reshape(B, T, self.head_dim)
        # O = Q^T (tied)
        attn_out = F.linear(attn_out, self.W_q.weight.t())  # [B, T, d_model]
        
        x = x + attn_out
        
        # ── FFN (SwiGLU) ──
        h = self.norm2(x)
        ffn = F.silu(self.W_gate(h)) * self.W_up(h)  # [B, T, ff_dim]
        ffn = self.W_down(ffn)  # [B, T, d_model]
        x = x + ffn
        
        # ── Output head ──
        x = self.norm_final(x)
        # LM head = embed.T (tied)
        logits = F.linear(x, self.embedding.weight)  # [B, T, V]
        
        return logits
    
    def param_count(self):
        """Count unique non-buffer parameters."""
        return sum(p.numel() for p in self.parameters())


# ── Data ──────────────────────────────────────────────────────────────────

def encode_pair(a, b):
    """Encode a+b in LSB-first reversed format: 0, a_rev, 0,0, b_rev, 0."""
    pa = f"{a:010d}"
    pb = f"{b:010d}"
    return [0] + [int(c) for c in reversed(pa)] + [0, 0] + [int(c) for c in reversed(pb)] + [0]
    #     1    10                                            2        10                   1 = 24 tokens


def expected_output(a, b):
    """Expected output digits in reversed order."""
    s = a + b
    return [int(d) for d in f"{s:011d}"[::-1]]  # 11 digits, LSB first


def generate_batch(batch_size, device, max_digits=10, 
                   hard_ratio=0.5, rng=None):
    """Generate training batch.
    
    Returns:
        full_seq: [B, 35] — 24 prompt + 11 output tokens
        labels: [B, 35] — -100 for prompt positions, output tokens for output
    """
    if rng is None:
        rng = np.random.RandomState()
    
    inputs = []
    labels = []
    
    for _ in range(batch_size):
        # Generate numbers up to max_digits
        max_val = 10 ** max_digits - 1
        a = rng.randint(0, max_val)
        b = rng.randint(0, max_val)
        
        inp = encode_pair(a, b)
        tgt = expected_output(a, b)
        
        # Teacher forcing: model sees prompt + predicts output
        full = inp + tgt  # 24 + 11 = 35
        lbl = [-100] * len(inp) + tgt  # mask prompt positions
        
        inputs.append(full)
        labels.append(lbl)
    
    return (
        torch.tensor(inputs, dtype=torch.long, device=device),
        torch.tensor(labels, dtype=torch.long, device=device),
    )


def generate_test_pairs(n, rng=None):
    """Generate n random (a,b) test pairs."""
    if rng is None:
        rng = np.random.RandomState(42)
    pairs = []
    for _ in range(n):
        a = rng.randint(0, 9999999999)
        b = rng.randint(0, 9999999999)
        pairs.append((a, b))
    return pairs


# ── Evaluation ────────────────────────────────────────────────────────────

def evaluate(model, device, n_samples=500, test_pairs=None, rng=None):
    """Evaluate exact-match accuracy on random pairs.
    
    Returns: (exact_accuracy, digit_accuracy)
    """
    if rng is None:
        rng = np.random.RandomState(42)
    
    if test_pairs is None:
        test_pairs = generate_test_pairs(n_samples, rng)
    
    model.eval()
    correct = 0
    total_digits_correct = 0
    total_digits = 0
    
    with torch.no_grad():
        for a, b in test_pairs[:n_samples]:
            exp = expected_output(a, b)
            inp = torch.tensor([encode_pair(a, b)], dtype=torch.long, device=device)
            
            # Autoregressive generation
            x = inp
            pred = []
            for _ in range(OUTPUT_DIGITS):
                logits = model(x)
                next_tok = logits[0, -1, :VOCAB_SIZE].argmax().item()
                pred.append(next_tok)
                x = torch.cat([x, torch.tensor([[next_tok]], device=device)], dim=1)
            
            if pred == exp:
                correct += 1
            for p, e in zip(pred, exp):
                if p == e:
                    total_digits_correct += 1
            total_digits += len(exp)
    
    return correct / len(test_pairs[:n_samples]), total_digits_correct / total_digits


def find_errors(model, device, test_pairs):
    """Find all (a,b) pairs the model gets wrong."""
    model.eval()
    errors = []
    with torch.no_grad():
        for a, b in test_pairs:
            exp = expected_output(a, b)
            inp = torch.tensor([encode_pair(a, b)], dtype=torch.long, device=device)
            x = inp
            pred = []
            for _ in range(OUTPUT_DIGITS):
                logits = model(x)
                next_tok = logits[0, -1, :VOCAB_SIZE].argmax().item()
                pred.append(next_tok)
                x = torch.cat([x, torch.tensor([[next_tok]], device=device)], dim=1)
            if pred != exp:
                errors.append((a, b))
    return errors


def build_targeted_batch(error_pairs, batch_size, device, rng):
    """Build a batch that includes all error pairs."""
    if len(error_pairs) >= batch_size:
        idxs = rng.choice(len(error_pairs), batch_size, replace=False)
        selected = [error_pairs[i] for i in idxs]
    else:
        selected = list(error_pairs)
    
    full_list, label_list = [], []
    for a, b in selected:
        inp = encode_pair(a, b)
        tgt = expected_output(a, b)
        full_list.append(inp + tgt)
        label_list.append([-100] * len(inp) + tgt)
    
    # Fill remaining with random pairs
    n_random = batch_size - len(full_list)
    if n_random > 0:
        rand_seq, rand_labels = generate_batch(n_random, device, max_digits=10, rng=rng)
        full_seq = torch.cat([torch.tensor(full_list, dtype=torch.long, device=device), rand_seq], dim=0)
        labels = torch.cat([torch.tensor(label_list, dtype=torch.long, device=device), rand_labels], dim=0)
    else:
        full_seq = torch.tensor(full_list, dtype=torch.long, device=device)
        labels = torch.tensor(label_list, dtype=torch.long, device=device)
    
    perm = torch.randperm(batch_size, device=device)
    return full_seq[perm], labels[perm]


# ── Training step ─────────────────────────────────────────────────────────

def train_step(model, full_seq, labels):
    """Single forward pass with teacher forcing.
    
    full_seq: [B, 35] — prompt + target tokens
    labels: [B, 35] — -100 for masked positions

    Returns: cross-entropy loss
    """
    logits = model(full_seq)  # [B, 35, 10]
    # Shift: logits[t] predicts token[t+1]
    shift_logits = logits[:, :-1, :].reshape(-1, VOCAB_SIZE)
    shift_labels = labels[:, 1:].reshape(-1)
    return F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train TinyAdder for FP16 quantization")
    parser.add_argument('--epochs', type=int, default=100000,
                        help='Training steps (batches)')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--eval-interval', type=int, default=2000)
    parser.add_argument('--eval-samples', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', default='retrain/weights')
    parser.add_argument('--cosine-lr', action='store_true', default=True,
                        help='Cosine LR decay from lr → lr/10')
    parser.add_argument('--warmup', type=int, default=0,
                        help='LR warmup steps')
    parser.add_argument('--targeted', action='store_true',
                        help='Targeted fine-tuning on error pairs')
    parser.add_argument('--targeted-steps', type=int, default=5000,
                        help='Steps for targeted fine-tuning')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--wandb', action='store_true')
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    device = torch.device(args.device)
    
    # Seed everything
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed)
    
    # Build model
    model = TinyAdderQwen3(d_model=4, head_dim=4, ff_dim=4, rope_theta=3.0)
    model.to(device)
    
    n_params = model.param_count()
    print(f"{'='*70}")
    print(f"TinyAdder Qwen3-style: d=4, hd=4, ff=4, RoPE θ=3, SwiGLU")
    print(f"Params: {n_params}")
    print(f"LR={args.lr}, batch={args.batch_size}, steps={args.epochs}, seed={args.seed}")
    print(f"{'='*70}")
    
    # Resume if requested
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        start_step = ckpt.get('step', 0)
        print(f"Resumed from {args.resume} (step {start_step}, acc {ckpt.get('acc', '?')})")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Test set for evaluation
    eval_rng = np.random.RandomState(999)
    eval_pairs = generate_test_pairs(2000, eval_rng)
    
    best_acc = 0.0
    t0 = time.time()
    
    # Metrics CSV
    metrics_path = os.path.join(args.output, 'metrics.csv')
    metrics_file = open(metrics_path, 'w', newline='')
    metrics_writer = csv.writer(metrics_file)
    metrics_writer.writerow(['step', 'loss', 'lr', 'exact_acc', 'digit_acc', 'elapsed'])
    
    # ── Training loop ──
    total_steps = args.targeted_steps if args.targeted else args.epochs
    mode = "Targeted FT" if args.targeted else "Training"
    
    for step in range(1, total_steps + 1):
        global_step = start_step + step
        
        # LR schedule
        if args.cosine_lr or args.warmup > 0:
            if step <= args.warmup:
                lr = args.lr * step / max(args.warmup, 1)
            elif args.cosine_lr and not args.targeted:
                progress = (step - args.warmup) / max(args.epochs - args.warmup, 1)
                lr = args.lr / 10 + 0.5 * (args.lr - args.lr / 10) * (1 + math.cos(math.pi * progress))
            else:
                lr = args.lr
            for pg in optimizer.param_groups:
                pg['lr'] = lr
        else:
            lr = args.lr
        
        model.train()
        
        # Batch generation
        if args.targeted:
            # Find errors and build targeted batch
            error_pairs = find_errors(model, device, eval_pairs[:1000])
            if not error_pairs:
                print("No errors found — model is already perfect!")
                break
            full_seq, labels = build_targeted_batch(error_pairs, args.batch_size, device, rng)
        else:
            full_seq, labels = generate_batch(args.batch_size, device, max_digits=10, rng=rng)
        
        optimizer.zero_grad()
        loss = train_step(model, full_seq, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Logging
        if step % 100 == 0:
            elapsed = time.time() - t0
            print(f"  step {global_step:6d} | loss {loss.item():.4f} | lr {lr:.2e} | {elapsed:.0f}s")
            metrics_writer.writerow([global_step, f"{loss.item():.6f}", f"{lr:.2e}", "", "", f"{elapsed:.1f}"])
            if step % 1000 == 0:
                metrics_file.flush()
        
        # Evaluation
        if step % args.eval_interval == 0:
            seq_acc, dig_acc = evaluate(model, device, test_pairs=eval_pairs[:min(500, len(eval_pairs))])
            elapsed = time.time() - t0
            n_errors = int((1 - seq_acc) * min(500, len(eval_pairs)))
            print(f"  EVAL step {global_step}: exact={seq_acc:.4f} ({n_errors} err) "
                  f"digit={dig_acc:.4f} [{elapsed:.0f}s]")
            
            metrics_writer.writerow([global_step, f"{loss.item():.6f}", f"{lr:.2e}", 
                                      f"{seq_acc:.4f}", f"{dig_acc:.4f}", f"{elapsed:.1f}"])
            metrics_file.flush()
            
            # Save best
            if seq_acc > best_acc:
                best_acc = seq_acc
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'step': global_step,
                    'acc': seq_acc,
                    'digit_acc': dig_acc,
                    'n_params': n_params,
                    'config': {
                        'd_model': 4, 'head_dim': 4, 'ff_dim': 4, 'rope_theta': 3.0,
                    },
                }, os.path.join(args.output, 'retrained_best.pt'))
                print(f"  ** NEW BEST: {seq_acc:.4f} **")
                
                # Also save FP16-quantized version
                sd_fp16 = {k: v.to(torch.float16).to(torch.float32) 
                           for k, v in model.state_dict().items()}
                torch.save({
                    'model_state_dict': sd_fp16,
                    'step': global_step,
                    'acc': seq_acc,
                    'digit_acc': dig_acc,
                    'n_params': n_params,
                }, os.path.join(args.output, 'retrained_best_fp16.pt'))
                
                # If perfect, stop early
                if seq_acc == 1.0:
                    print(f"  *** PERFECT ACCURACY achieved at step {global_step}! ***")
                    break
    
    metrics_file.close()
    
    # Final evaluation
    print(f"\n{'='*70}")
    seq_acc, dig_acc = evaluate(model, device, n_samples=2000, test_pairs=eval_pairs)
    print(f"FINAL: exact={seq_acc:.4f} digit={dig_acc:.4f} params={n_params}")
    print(f"Best: {best_acc:.4f}")
    print(f"Total time: {time.time() - t0:.0f}s")
    
    # Save final
    torch.save({
        'model_state_dict': model.state_dict(),
        'step': args.epochs,
        'acc': seq_acc,
        'digit_acc': dig_acc,
        'n_params': n_params,
    }, os.path.join(args.output, 'retrained_final.pt'))
    
    # FP16 evaluation
    model_fp16 = TinyAdderQwen3(d_model=4, head_dim=4, ff_dim=4, rope_theta=3.0)
    model_fp16.to(device)
    sd_fp16 = {k: v.to(torch.float16).to(torch.float32) 
               for k, v in model.state_dict().items()}
    model_fp16.load_state_dict(sd_fp16)
    fp16_seq_acc, fp16_dig_acc = evaluate(model_fp16, device, test_pairs=eval_pairs[:2000])
    print(f"\nFP16 quantized: exact={fp16_seq_acc:.4f} digit={fp16_dig_acc:.4f}")
    
    # Weight analysis
    print(f"\nWeight range analysis:")
    for name, p in model.named_parameters():
        p_np = p.detach().numpy()
        fp16_np = p.detach().to(torch.float16).to(torch.float32).numpy()
        diff = np.abs(p_np - fp16_np).max()
        print(f"  {name:25s}: range=[{p_np.min():.2f}, {p_np.max():.2f}], "
              f"fp16_err={diff:.2e}")


if __name__ == '__main__':
    main()
