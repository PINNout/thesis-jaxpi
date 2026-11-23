"""
Add the precision hook to each example's train.py (safe to run repeatedly).

  python3 patch_train.py            # patches the examples listed below

Adds:  from jaxpi.precision import set_precision      (before 'import models')
       set_precision(config.get("precision"))         (first line of train_and_evaluate)
"""
import os, re

EXAMPLES = ["adv", "allen_cahn", "burgers", "ks", "ldc", "ns_steady_cylinder"]
ROOT = os.environ.get("JAXPI_EXAMPLES", os.path.expanduser("~/thesis/jaxpi/examples"))

for ex in EXAMPLES:
    path = os.path.join(ROOT, ex, "train.py")
    if not os.path.isfile(path):
        print(f"{ex:20s} no train.py, skipped")
        continue
    src = open(path).read()

    # KS: let the config choose how many of the time windows get trained
    if ex == "ks" and "windows_to_train" not in src:
        new_src, n = re.subn(
            r"for idx in range\(config\.training\.num_time_windows\):",
            'for idx in range(config.training.get("windows_to_train", config.training.num_time_windows)):',
            src, count=1)
        if n == 1:
            src = new_src
            open(path, "w").write(src)
            print(f"{ex:20s} windows_to_train patch added")
        else:
            print(f"{ex:20s} COULD NOT add windows_to_train patch - edit the window loop by hand")

    if "set_precision" in src:
        print(f"{ex:20s} already patched")
        continue
    src, n1 = re.subn(r"^import models$", "from jaxpi.precision import set_precision\n\nimport models", src, count=1, flags=re.M)
    src, n2 = re.subn(r"(def train_and_evaluate\([^)]*\):\n)", r'\1    set_precision(config.get("precision"))\n\n', src, count=1)
    if n1 != 1 or n2 != 1:
        print(f"{ex:20s} COULD NOT PATCH (import: {n1}, def: {n2}) - edit by hand")
        continue
    open(path, "w").write(src)
    print(f"{ex:20s} patched")
