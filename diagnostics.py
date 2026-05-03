"""diagnostics.py — find the source of training NaNs by reproducing the
failing config under torch.autograd.set_detect_anomaly() with hook-based
inspection of forward + backward.

Specifically targets the levels=11 / lr=0.01 / bs=16384 / wavelet_crawl=False
NaN that appears at step ~500. The script:

  1. Builds the model with the failing config + deterministic seed (1337)
  2. Trains step by step, saving a checkpoint at --start_save_step (just
     before the suspected NaN onset) so the failure can be re-reproduced
  3. From --start_anomaly_step onward, enables torch.autograd.set_detect_anomaly()
     so any NaN/Inf in the BACKWARD pass raises with a stack trace pointing
     at the offending op
  4. Per step: after loss.backward(), checks every parameter's gradient for
     finiteness; on NaN, captures full state (weights, optimizer state, the
     offending batch, plus forward-hook info on which module first emitted
     non-finite output) into a debug dir
  5. After the diagnostic prints, deletes the debug files it created (use
     --keep_debug to preserve them for further inspection)

Usage:
    python diagnostics.py
    python diagnostics.py --levels 9 --start_save_step 1700
    python diagnostics.py --keep_debug                    # don't auto-delete state

Defaults reproduce the worst observed NaN: levels=11 at lr=0.01 (NaN at step ~500).
"""
import argparse
import os
import sys

import torch
import torch.nn as nn

import train as train_mod
from model import WaveletLM, set_seed


# ============================================================================
# Tensor / module helpers
# ============================================================================

def first_nonfinite_tensor(t):
    if isinstance(t, torch.Tensor):
        if not torch.isfinite(t).all().item():
            return t
        return None
    if isinstance(t, (tuple, list)):
        for x in t:
            r = first_nonfinite_tensor(x)
            if r is not None:
                return r
    return None


def tensor_summary(t):
    if not isinstance(t, torch.Tensor):
        return None
    finite = torch.isfinite(t)
    if finite.any():
        max_abs = float(t[finite].abs().max().item())
    else:
        max_abs = float('inf')
    nonfin = int((~finite).sum().item())
    return {
        'shape': tuple(t.shape),
        'dtype': str(t.dtype),
        'max_abs_finite': max_abs,
        'nonfinite_count': nonfin,
        'total': t.numel(),
    }


class NanForwardCatcher:
    """Forward-hook detector — captures the FIRST module (in execution order)
    that emits a non-finite output during the forward pass."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.handles = []
        self.first_failure = None
        self.module_names = {id(m): n for n, m in model.named_modules()}

    def reset(self):
        self.first_failure = None

    def install(self):
        def make_hook(mod):
            name = self.module_names.get(id(mod), type(mod).__name__)

            def hook(_m, inputs, output):
                if self.first_failure is not None:
                    return
                bad = first_nonfinite_tensor(output)
                if bad is None:
                    return
                in_summaries = []
                seq = inputs if isinstance(inputs, (tuple, list)) else (inputs,)
                for i, t in enumerate(seq):
                    if isinstance(t, torch.Tensor):
                        s = tensor_summary(t)
                        s['arg_idx'] = i
                        in_summaries.append(s)
                self.first_failure = {
                    'module_name': name,
                    'module_class': type(mod).__name__,
                    'output': tensor_summary(bad),
                    'inputs': in_summaries,
                }
            return hook

        for mod in self.model.modules():
            self.handles.append(mod.register_forward_hook(make_hook(mod)))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


# ============================================================================
# Config builder — replicates the failing levels=11 sweep iteration
# ============================================================================

def build_failing_config(levels: int, lr: float, min_lr: float,
                         no_compile: bool = True) -> dict:
    """Build a config matching the levels=N retry from runs.sh Phase 2."""
    S = levels + 1
    half = S // 2
    psmw = [1.0] * half + [0.5] * half
    return {
        'dataset': 'wikitext-103',
        'tokenizer': 'auto',
        'out_dir': 'logs',
        # Disable torch.compile so anomaly-mode stack traces point at the
        # actual model code, not a generated graph.
        'compile': not no_compile,
        'compile_mode': 'default',
        'seed': 1337,
        'epochs': 1,
        'micro_batch_size': 1,
        'grad_accum': 1,
        'block_size': 16384,
        'eval_interval': 250,
        'skip_warmup_saves': True,
        'C': 2048,
        'layers': 1,
        'levels': levels,
        'low_rank': 4,
        'mlp_expansion': 10,
        'mlp_layers': 2,
        'pkm_enabled': False,
        'pkm_num_keys': 16384,
        'pkm_top_k': 32,
        'pkm_heads': 1,
        'fwpkm_enabled': True,
        'fwpkm_num_keys': 8281,
        'fwpkm_top_k': 32,
        'fwpkm_heads': 1,
        'fwpkm_inference_updates': False,
        'fwpkm_update_lr': 0.01,
        'fwpkm_chunk_size': 64,
        'wavelet_mode': 'lifting',
        'shared_lifting_weights': True,
        'untied_reconstruction': False,
        'multi_basis_lifting': False,
        'multi_basis_inits': ['haar', 'random'],
        'cross_scale_gating': True,
        'per_scale_mixer_widths': psmw,
        'wavelet_crawl': False,
        'wavelet_crawl_k': 3,
        'lifting_linear_only': False,
        'lifting_hidden_mult': 1,
        'lifting_init': 'haar',
        'lifting_dropout': 0.0,
        'use_mixer_gate': True,
        'mixer_gate_activation': 'silu',
        'mixer_depth': 1,
        'mixer_depth_stabilizers': False,
        'mixer_depth_residuals': False,
        'decompose_bypass': True,
        'decompose_bypass_cross_window': True,
        'decompose_bypass_ema': False,
        'learned_residual': True,
        'skip_proj_out': False,
        'looped_blocks': False,
        'looped_blocks_count': 8,
        'stochastic_depth_rate': 0.0,
        'dropout_embedding': 0.2,
        'dropout_projection': 0.1,
        'dropout_mixer': 0.1,
        'dropout_mlp': 0.1,
        'dropout_lm_head': 0.24,
        'optimizer': 'Adagrad',
        'optimizer_eps': 2e-13,
        'lr': lr,
        'min_lr': min_lr,
        'warmup_fraction': 0.3,
        'grad_clip': 1.0,
        'tie_embedding_to_lm_head': True,
        'gradient_checkpointing': False,
        'use_amp': True,
        'amp_dtype': 'fp16',
        'allow_tf32': True,
        'per_layer_embedding': True,
        'loop_iterations': 1,
        'weight_decay': 1e-06,
        'stable_parametrization': False,
        'stab_spectral_norm': False,
        'stab_ff_scaling': False,
        'stab_embed_scaling': False,
        'stab_proj_out_scaling': False,
        'stab_mixer_eps_scaling': False,
        'stab_lifting_level_scaling': False,
        'multinodal_enabled': False,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--levels', type=int, default=11)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--min_lr', type=float, default=0.0002)
    ap.add_argument('--start_save_step', type=int, default=470,
                    help='Save a pre-NaN checkpoint at this step (default: 470, ~30 steps before known levels=11 NaN)')
    ap.add_argument('--start_anomaly_step', type=int, default=470,
                    help='Enable torch.autograd.set_detect_anomaly() at this step')
    ap.add_argument('--max_steps', type=int, default=600,
                    help='Stop after this many steps if no NaN seen')
    ap.add_argument('--debug_dir', default='logs/diagnostics_debug',
                    help='Where to save debug state')
    ap.add_argument('--keep_debug', action='store_true',
                    help='Preserve debug files after diagnosis (default: delete to save disk)')
    ap.add_argument('--enable_compile', action='store_true',
                    help='Run with torch.compile (default: off — clearer stack traces)')
    args = ap.parse_args()

    config = build_failing_config(args.levels, args.lr, args.min_lr,
                                  no_compile=not args.enable_compile)

    print(f"=== diagnostics.py — NaN root-cause sweep ===")
    print(f"  Config: levels={args.levels}, lr={args.lr}, min_lr={args.min_lr}, "
          f"compile={config['compile']}")
    print(f"  Save pre-NaN checkpoint at step: {args.start_save_step}")
    print(f"  Enable anomaly mode at step:     {args.start_anomaly_step}")
    print(f"  Hard cap on steps:               {args.max_steps}")
    print(f"  Debug dir:                       {args.debug_dir}")
    print(f"  Keep debug files after run:      {args.keep_debug}")
    print()

    set_seed(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        sys.exit("CUDA required for diagnostics — model + 16K-context tensors are too large for CPU.")

    # Load dataset (silent — we only care about training)
    class _SilentLogger:
        def log(self, _msg): pass
    train_data, val_data, _test, enc, _bpb = train_mod.load_and_encode_dataset(
        config, _SilentLogger())
    vocab_size = enc.vocab_size

    # Build model
    print(f"  Building model (vocab_size={vocab_size})...")
    model = WaveletLM(vocab_size=vocab_size, config=config, device=device).to(device)
    if config['compile']:
        model = torch.compile(model)
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params/1e6:.2f}M params\n")

    # Optimizer — match train.py's Adagrad config
    optimizer = torch.optim.Adagrad(
        model.parameters(),
        lr=config['lr'],
        eps=config['optimizer_eps'],
        weight_decay=config['weight_decay'],
    )

    # AMP / scaler
    use_amp = config['use_amp']
    amp_dtype = torch.float16 if config['amp_dtype'] == 'fp16' else torch.bfloat16
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and amp_dtype == torch.float16))

    # Data
    get_batch = train_mod.make_get_batch(train_data, val_data, config, device)

    # LR schedule (replicate train.py:797-800)
    T = config['block_size']
    effective_batch = config['micro_batch_size'] * config.get('grad_accum', 1)
    steps_per_epoch = len(train_data) // (T * effective_batch)
    total_steps = config['epochs'] * steps_per_epoch
    warmup_steps = int(config['warmup_fraction'] * total_steps)
    print(f"  Schedule: steps_per_epoch={steps_per_epoch}, total_steps={total_steps}, "
          f"warmup_steps={warmup_steps}\n")

    # Debug dir setup
    os.makedirs(args.debug_dir, exist_ok=True)
    debug_files_created = []

    def cleanup():
        if args.keep_debug:
            print(f"  --keep_debug set; preserving files in {args.debug_dir}/")
            return
        deleted = 0
        for f in debug_files_created:
            if os.path.isfile(f):
                try:
                    os.remove(f)
                    deleted += 1
                except OSError:
                    pass
        if os.path.isdir(args.debug_dir) and not os.listdir(args.debug_dir):
            try:
                os.rmdir(args.debug_dir)
            except OSError:
                pass
        print(f"  Deleted {deleted} debug file(s) from {args.debug_dir}/")

    catcher = None
    nan_found_step = None

    try:
        for step in range(args.max_steps):
            X, Y = get_batch('train')

            # Apply LR schedule
            lr = train_mod.get_lr(step, config, total_steps, warmup_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            # At the configured step, save a pre-NaN checkpoint and arm
            # the anomaly-detection machinery
            if step == args.start_save_step:
                ckpt_path = os.path.join(args.debug_dir, f'pre_nan_step{step}.pt')
                torch.save({
                    'step': step,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scaler_state': scaler.state_dict() if scaler else None,
                }, ckpt_path)
                debug_files_created.append(ckpt_path)
                print(f"  [step {step}] Saved pre-NaN checkpoint: {ckpt_path}")

                catcher = NanForwardCatcher(model)
                catcher.install()
                torch.autograd.set_detect_anomaly(True)
                print(f"  [step {step}] Forward hooks installed + autograd anomaly mode enabled")

            # Reset hook state at the start of every armed step so the catcher
            # reports per-step rather than just the very first failure
            if catcher is not None:
                catcher.reset()

            # Forward
            with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                _logits, loss = model(X, Y)

            # If forward already produced a non-finite loss, the model itself
            # blew up — capture state and stop
            if not torch.isfinite(loss).item():
                print(f"\n  [step {step}] Loss is non-finite (loss.item()={loss.item()}).")
                if catcher is not None and catcher.first_failure:
                    f = catcher.first_failure
                    out = f['output']
                    print(f"    First module to emit non-finite forward output:")
                    print(f"      module_name:   {f['module_name']}")
                    print(f"      module_class:  {f['module_class']}")
                    print(f"      output shape:  {out['shape']} dtype={out['dtype']}")
                    print(f"      output max|finite|: {out['max_abs_finite']:.4g}")
                    print(f"      output non-finite: {out['nonfinite_count']:,}/{out['total']:,}")
                    for s in f['inputs']:
                        print(f"      input arg[{s['arg_idx']}]: shape={s['shape']} "
                              f"max|finite|={s['max_abs_finite']:.4g} "
                              f"non-finite={s['nonfinite_count']:,}/{s['total']:,}")
                else:
                    print(f"    No forward hook fired — non-finite originated outside any captured module "
                          f"(possibly in loss computation or AMP scaler).")

                # Save the offending batch + state
                fail_ckpt = os.path.join(args.debug_dir, f'nan_state_step{step}.pt')
                batch_path = os.path.join(args.debug_dir, f'failing_batch_step{step}.pt')
                torch.save({
                    'step': step,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scaler_state': scaler.state_dict() if scaler else None,
                }, fail_ckpt)
                torch.save({'X': X.cpu(), 'Y': Y.cpu(), 'step': step}, batch_path)
                debug_files_created.extend([fail_ckpt, batch_path])
                nan_found_step = step
                break

            # Backward (anomaly mode raises with a stack trace if a NaN/Inf
            # appears in any backward op; the try/except below preserves the
            # diagnostic message)
            try:
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            except RuntimeError as e:
                # Anomaly mode raised — print the stack trace info verbatim
                print(f"\n  [step {step}] Anomaly-mode backward exception:")
                print(f"  {e}")
                if catcher is not None and catcher.first_failure:
                    f = catcher.first_failure
                    print(f"\n  Forward-hook captured this BEFORE the backward exception:")
                    print(f"    First module to emit non-finite forward output: "
                          f"{f['module_name']} ({f['module_class']})")
                # Save state
                fail_ckpt = os.path.join(args.debug_dir, f'nan_state_step{step}.pt')
                batch_path = os.path.join(args.debug_dir, f'failing_batch_step{step}.pt')
                torch.save({
                    'step': step,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                }, fail_ckpt)
                torch.save({'X': X.cpu(), 'Y': Y.cpu(), 'step': step}, batch_path)
                debug_files_created.extend([fail_ckpt, batch_path])
                nan_found_step = step
                break

            # Check parameter gradients for NaN
            nan_param_names = []
            for name, p in model.named_parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all().item():
                    nan_param_names.append(name)

            if nan_param_names:
                print(f"\n  [step {step}] NaN/Inf in gradients of {len(nan_param_names)} param(s). First 10:")
                params_dict = dict(model.named_parameters())
                for name in nan_param_names[:10]:
                    s = tensor_summary(params_dict[name].grad)
                    print(f"    {name}: shape={s['shape']} "
                          f"non-finite={s['nonfinite_count']:,}/{s['total']:,} "
                          f"max|finite|={s['max_abs_finite']:.4g}")

                if catcher is not None and catcher.first_failure:
                    f = catcher.first_failure
                    out = f['output']
                    print(f"\n  Forward-hook captured during this same step:")
                    print(f"    First module to emit non-finite forward output: "
                          f"{f['module_name']} ({f['module_class']})")
                    print(f"    Output shape={out['shape']} max|finite|={out['max_abs_finite']:.4g} "
                          f"non-finite={out['nonfinite_count']:,}/{out['total']:,}")
                else:
                    print(f"\n  Forward was clean — NaN appeared in backward only.")

                # Save state
                fail_ckpt = os.path.join(args.debug_dir, f'nan_state_step{step}.pt')
                batch_path = os.path.join(args.debug_dir, f'failing_batch_step{step}.pt')
                torch.save({
                    'step': step,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                }, fail_ckpt)
                torch.save({'X': X.cpu(), 'Y': Y.cpu(), 'step': step}, batch_path)
                debug_files_created.extend([fail_ckpt, batch_path])
                nan_found_step = step
                break

            # Normal optimizer step
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # Periodic progress
            if step % 50 == 0 or step >= args.start_save_step:
                print(f"  [step {step:>4}] loss={loss.item():.4f}, lr={lr:.4e}")
        else:
            print(f"\n  Reached --max_steps={args.max_steps} without observing NaN.")
            print(f"  Increase --max_steps or lower --start_save_step to widen the search window.")
    finally:
        if catcher is not None:
            catcher.remove()
        torch.autograd.set_detect_anomaly(False)
        cleanup()

    if nan_found_step is not None:
        print(f"\n=== NaN diagnosed at step {nan_found_step} ===")


if __name__ == '__main__':
    main()
