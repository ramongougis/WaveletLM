"""perf_probe.py — why is C=256 no faster than C=512 on this pod?

MEASURED FACTS THIS EXISTS TO EXPLAIN (2026-07-28):
  * D0 config (C=512): 2.27 it/s on the 07-12 pod, 1.30 it/s on this pod (identical code).
  * Micro (C=256) on this pod: 1.31 it/s — a model with 2.9x fewer params, SAME speed.
  * Telemetry during both: CPU 100%, GPU util 13%, 163W/575W.
  * Compute-bound floor would be ~9 it/s (C=512) / ~27 it/s (C=256) at 50 TFLOPS.

THE QUESTIONS, MAPPED TO SECTIONS:
  [A] environment census   — is the box what it claims? (cgroup quota vs visible cores,
                             torch's thread-pool sizing)
  [B] launch rate          — how many CUDA kernels/second can THIS CPU dispatch, and does
                             capping threads change it? Time/step ~= N_kernels x t_launch
                             when dispatch-bound, which is C-INDEPENDENT — the identical
                             C=256/C=512 timings in one number.
  [C] GPU health           — big-matmul TFLOPS. If low, the GPU itself is throttled and
                             the pod should be abandoned, not tuned.
  [D] the real model       — eager step time + profiler kernel count at C=256 AND C=512.
                             Predicted dispatch time = count x t_launch; if that matches
                             measured, the diagnosis is closed.

THE OVERSUBSCRIPTION HYPOTHESIS (what [A]+[B] test): this pod shows ~120 visible cores
but enforces a ~12.75-core cgroup quota. PyTorch sizes OMP/intra-op pools from VISIBLE
cores at import, and idle OMP threads SPIN-WAIT — burning quota doing nothing. When the
quota exhausts inside a 100ms cgroup period the kernel freezes EVERY thread until the
next period, including the one dispatching CUDA kernels. Periodic whole-process stalls,
GPU starves, CPU reads 100%.

USAGE (on the pod; PAUSE THE TRAINING QUEUE FIRST — a concurrent run skews everything):
    python tools/perf_probe.py                       # default thread env
    OMP_NUM_THREADS=4 python tools/perf_probe.py     # capped BEFORE import (spin fix)
Compare the two outputs side by side. The env-var run is the one that can catch the
spin-wait mechanism; set_num_threads() inside a live process cannot fully undo it.
"""
import argparse, json, os, sys, time
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)


def census():
    print("=" * 70)
    print("[A] ENVIRONMENT CENSUS")
    n = os.cpu_count()
    print(f"    visible CPUs (os.cpu_count)  : {n}")
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        eff = "unlimited" if quota == "max" else f"{int(quota)/int(period):.2f} cores"
        print(f"    cgroup cpu.max               : {quota} {period}  -> {eff}")
        if quota != "max" and n and int(quota)/int(period) < n / 4:
            print(f"    *** OVERSUBSCRIPTION RISK: pools sized for {n} cores, "
                  f"quota is {eff} ***")
    except OSError:
        print("    cgroup cpu.max               : (not readable — not cgv2/linux)")
    print(f"    torch.get_num_threads()      : {torch.get_num_threads()}")
    print(f"    OMP_NUM_THREADS env          : {os.environ.get('OMP_NUM_THREADS', '(unset)')}")
    try:
        model = [l.split(":")[1].strip() for l in open("/proc/cpuinfo")
                 if l.startswith("model name")]
        print(f"    CPU                          : {model[0]}  (x{len(model)} visible)")
    except OSError:
        pass
    print(f"    GPU                          : {torch.cuda.get_device_name(0)}")
    print(f"    torch {torch.__version__}, cuda {torch.version.cuda}")


def launch_rate(n=20000):
    print("=" * 70)
    print("[B] KERNEL LAUNCH RATE (tiny elementwise op — pure dispatch, no compute)")
    x = torch.ones(256, device="cuda")
    for _ in range(200):
        x.add_(1.0)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        x.add_(1.0)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    us = dt / n * 1e6
    print(f"    {n:,} launches in {dt:.2f}s  ->  {n/dt:,.0f} launches/s  "
          f"({us:.1f} us/launch)")
    print("    healthy dedicated box: 3-8 us/launch. 20+ us = CPU-side trouble")
    print("    (slow core, quota throttling, or oversubscription stalls).")
    return us


def matmul_tflops():
    print("=" * 70)
    print("[C] GPU HEALTH (fp16 4096^3 matmul — no dispatch pressure, one big kernel)")
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    for _ in range(5):
        a @ b
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        a @ b
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    tf = 2 * 4096**3 * 50 / dt / 1e12
    print(f"    {tf:,.0f} TFLOPS sustained fp16")
    print("    RTX 5090 should land well over 100. If it does and training still")
    print("    crawls, the GPU is fine and the bottleneck is upstream of it.")
    return tf


def model_probe(run_dir, C, us_per_launch, steps=12, mbs=None):
    """mbs=None uses the config's own batch. On OOM the caller retries at half batch —
    EAGER mode holds more intermediates than compiled training (no fusion), so a config
    that trains fine at MBS=48 compiled can OOM here (measured: C=512 eager needs >31GB
    at MBS=48, 2026-07-28). Kernel COUNT is batch-independent (same graph), so the
    dispatch census stays valid at any MBS; only ms/step loses direct comparability."""
    import gc
    from model import WaveletLM
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    cfg.update(C=C, compile=False, device="cuda")
    if mbs is not None:
        cfg["micro_batch_size"] = mbs
    torch.manual_seed(0)
    m = WaveletLM(50257, cfg, device="cuda").to("cuda")
    m.train()
    opt = torch.optim.Adagrad(m.parameters(), lr=1e-4, eps=2e-13)
    scaler = torch.amp.GradScaler("cuda")
    x = torch.randint(0, 50257, (cfg["micro_batch_size"], cfg["block_size"]), device="cuda")
    y = torch.randint(0, 50257, (cfg["micro_batch_size"], cfg["block_size"]), device="cuda")

    def step():
        with torch.autocast("cuda", dtype=torch.float16):
            _, loss = m(x, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        scaler.step(opt); scaler.update()

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        step()
    torch.cuda.synchronize()
    per = (time.perf_counter() - t0) / steps

    # kernel census for ONE step
    kcount = None
    try:
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            step()
            torch.cuda.synchronize()
        kcount = sum(ka.count for ka in prof.key_averages()
                     if ka.self_device_time_total > 0)
    except Exception as e:
        print(f"    (profiler unavailable: {type(e).__name__})")

    print(f"    C={C:4d}: {per*1000:7.0f} ms/step  ({1/per:.2f} it/s, eager)", end="")
    if kcount:
        pred = kcount * us_per_launch / 1e6
        print(f"   kernels/step ~{kcount:,}  dispatch-only prediction "
              f"{pred*1000:.0f} ms ({100*pred/per:.0f}% of measured)")
    else:
        print()
    del m, opt, scaler
    gc.collect()
    torch.cuda.empty_cache()
    return per, kcount


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="logs/wikitext-103_2026-07-12_08-04-38",
                    help="config source (D0 by default)")
    ap.add_argument("--skip_model", action="store_true")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required — run this on the pod, not the workstation.")

    census()
    us = launch_rate()
    matmul_tflops()
    if not args.skip_model:
        print("=" * 70)
        print("[D] THE REAL MODEL (eager, fwd+bwd+Adagrad, MBS/block from D0 config)")
        def probe_with_fallback(C):
            mbs = None
            for _ in range(3):
                try:
                    return model_probe(args.run_dir, C, us, mbs=mbs)
                except torch.OutOfMemoryError:
                    cfg_mbs = json.load(open(os.path.join(
                        args.run_dir, "config.json")))["micro_batch_size"]
                    mbs = (mbs or cfg_mbs) // 2
                    torch.cuda.empty_cache()
                    print(f"    C={C}: OOM in eager mode, retrying at MBS={mbs} "
                          f"(kernel count unaffected; ms/step not comparable to training)")
            return None, None
        p256, k256 = probe_with_fallback(256)
        p512, k512 = probe_with_fallback(512)
        if p256 is None or p512 is None:
            print("    (a probe arm failed even at reduced MBS)")
            return
        print(f"    C=512 / C=256 step-time ratio: {p512/p256:.2f}x "
              f"(compute-bound would be ~2.6x; dispatch-bound is ~1.0x)")
        if k256 and k512:
            same = abs(k256 - k512) < 0.1 * max(k256, k512)
            verdict = ("~equal — count is C-independent, as hypothesized" if same
                       else "DIFFER — revisit the hypothesis")
            print(f"    kernel counts: C=256 {k256:,} vs C=512 {k512:,}  ({verdict})")
    print("=" * 70)
    print("READ-OUT:")
    print("  * [C] low                        -> pod GPU throttled; change pods, done.")
    print("  * [B] slow + [D] prediction fits -> dispatch-bound confirmed. Rerun this")
    print("    probe with OMP_NUM_THREADS=4: if [B] improves a lot, it's the spin-wait/")
    print("    oversubscription mechanism -> set the env var in runs.sh and retest")
    print("    training; also consider compile_mode='reduce-overhead' (CUDA graphs)")
    print("    which replays one captured graph instead of thousands of launches.")
    print("  * [B] fine but training slow     -> the gap is in the training loop, not")
    print("    dispatch (eval tax was one such; already fixed).")


if __name__ == "__main__":
    main()
