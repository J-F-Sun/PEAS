"""Regenerate the three paper PDFs from archived raw predictions."""
import argparse,os
from pathlib import Path
from peas.common.paths import repository_root

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,default=repository_root()/"outputs/figures");p.add_argument("--ablation-version",choices=("current","previous"),default="current");a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);os.environ.setdefault("MPLCONFIGDIR",str(out/".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from peas.evaluation import probability_plot as prob, relative_plot as rel
    plt.rcParams.update({"font.family":"serif","font.serif":["STIXGeneral","DejaVu Serif"],"mathtext.fontset":"stix","font.size":8,"axes.titlesize":8.5,"axes.labelsize":8.25,"xtick.labelsize":7.5,"ytick.labelsize":7.25,"axes.linewidth":.65,"axes.spines.top":False,"axes.spines.right":False,"legend.fontsize":7.9,"legend.frameon":False,"pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none"})
    prob.OUT=out;prob.render(plt,prob.load_curves()[0],with_errors=True)
    plt.rcParams.update({"axes.labelsize":8.5,"xtick.labelsize":7.4})
    rel.AUDIT=out
    rel.render(plt,rel.load_scores(rel.COMPARISON,comparison=True),rel.COMPARISON,out/"exp2_relative_l2_positions_and_mean_1x3_new",stacked=False)
    rel.render(plt,rel.load_scores(rel.ABLATIONS,comparison=False,ablation_version=a.ablation_version),rel.ABLATIONS,out/"exp3_ablation_relative_l2_positions_and_mean_1x3",stacked=False)
    print(out)
if __name__=="__main__":main()
