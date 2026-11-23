"""
Drop-in replacement for main.py that opens the CUDA profiler window only
around a few steady-state training steps, so Nsight Systems / Nsight Compute
skip the JIT compilation and the setup.

Run from inside an example folder with a PI_PROFILE=1 config, e.g.
    PI_PROFILE=1 python3 profile_main.py --config=benchmark_configs/burgers_fp32.py --workdir=./profiles/tmp
and start it under nsys / ncu with the profiler-API capture mode
(see run_profiles.sh). The profile configs run 60 steps and log every 40:
  step 0   -> logged (JIT is done)                  => profiler START
  step 40  -> evaluator entered (before its work)   => profiler STOP
so exactly training steps 1..40 are captured, without evaluation kernels.
"""
import ctypes, glob, os, sys

sys.path.insert(0, os.getcwd())   # so `import main` finds the example's main.py


def _load_cudart():
    names = ["libcudart.so", "libcudart.so.12", "libcudart.so.13"]
    for n in names:
        try:
            return ctypes.CDLL(n)
        except OSError:
            pass
    import site
    for sp in site.getsitepackages():
        for p in glob.glob(os.path.join(sp, "nvidia", "**", "libcudart.so*"), recursive=True):
            try:
                return ctypes.CDLL(p)
            except OSError:
                pass
    raise RuntimeError("could not find libcudart (needed for cudaProfilerStart/Stop)")


class _Dry:  # PI_FAKE_CUDART=1: dry run without a GPU, just prints
    def cudaProfilerStart(self): print("(dry) profiler start")
    def cudaProfilerStop(self): print("(dry) profiler stop")


_cudart = _Dry() if os.environ.get("PI_FAKE_CUDART") else _load_cudart()
_calls = {"n": 0}

from jaxpi import logging as _pl
from jaxpi import evaluator as _ev

# Window: opens after the first log (step 0, JIT already done) and closes when
# the evaluator is entered for the second time (step 40), BEFORE it computes its
# diagnostics, so only pure training steps 1..40 are captured.
_orig_log = _pl.Logger.log_iter
_orig_eval = _ev.BaseEvaluator.__call__
_state = {"started": False, "stopped": False, "evals": 0}


def _log_iter(self, step, start_time, end_time, log_dict):
    _orig_log(self, step, start_time, end_time, log_dict)
    if not _state["started"]:
        import jax
        jax.effects_barrier()
        _state["started"] = True
        print(">>> cudaProfilerStart", flush=True)
        _cudart.cudaProfilerStart()


def _eval_call(self, *args, **kwargs):
    _state["evals"] += 1
    if _state["started"] and not _state["stopped"] and _state["evals"] >= 2:
        import jax
        jax.effects_barrier()
        _cudart.cudaProfilerStop()
        _state["stopped"] = True
        print(">>> cudaProfilerStop", flush=True)
    return _orig_eval(self, *args, **kwargs)


_pl.Logger.log_iter = _log_iter
_ev.BaseEvaluator.__call__ = _eval_call

from absl import app, flags
import main  # the example's own main.py

if __name__ == "__main__":
    flags.mark_flags_as_required(["config", "workdir"])
    app.run(main.main)
