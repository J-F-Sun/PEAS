"""Recompute per-position/overall errors, sample SD, and confidence intervals."""
import argparse
from pathlib import Path
from peas.common.paths import repository_root
from peas.evaluation.metrics import summarize,write_metrics
p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,default=repository_root()/"outputs/metrics.csv");p.add_argument("--bootstrap",type=int,default=2048)
a=p.parse_args();rows=summarize(a.bootstrap);write_metrics(a.output,rows)
for r in rows:
    if r["id"]=="__mean__":print(r["group"],r["scene"],r["method"],f'{r["mean_percent"]:.6f}%')
print(a.output)
