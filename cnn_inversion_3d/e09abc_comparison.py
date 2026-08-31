"""Compare corrected E09A, E09B, and E09C on identical held-out samples."""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from cnn_inversion_3d import failure_mode_analysis as analysis
def main():
    p=argparse.ArgumentParser(); p.add_argument("--dataset",type=Path,default=Path("datasets/canonical_single_plane_train2000"))
    p.add_argument("--e09a",type=Path,default=Path("prediction_outputs/E09A_depth_supervision_corrected")); p.add_argument("--e09b",type=Path,default=Path("prediction_outputs/E09B_integrated_sensitivity")); p.add_argument("--e09c",type=Path,default=Path("prediction_outputs/E09C_canonical_single_plane_extent_loss_bf8")); p.add_argument("--output",type=Path,default=Path("analysis_outputs/E09A_E09B_E09C_comparison")); a=p.parse_args()
    analysis.EXPERIMENTS={"E09A":str(a.e09a),"E09B":str(a.e09b),"E09C":str(a.e09c)}; out=a.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    rows,by=analysis.collect(Path.cwd(),a.dataset.resolve()); overall=analysis.aggregate(by); analysis.write_csv(out/"overall_comparison.csv",overall); analysis.write_csv(out/"sample_metrics.csv",rows)
    metrics=(("mean_iou","Mean IoU"),("mean_dice","Mean Dice"),("mean_absolute_top_depth_error_m","Top-depth MAE (m)"),("mean_absolute_bottom_depth_error_m","Bottom-depth MAE (m)"),("mean_absolute_thickness_error_m","Thickness MAE (m)"))
    fig,axes=plt.subplots(1,5,figsize=(17,4),constrained_layout=True); labels=[r["experiment"] for r in overall]
    for ax,(key,title) in zip(axes,metrics): ax.bar(labels,[r[key] for r in overall]); ax.set_title(title); ax.grid(axis="y",alpha=.25)
    fig.savefig(out/"e09abc_metrics.png",dpi=180); plt.close(fig); print(*overall,sep="\n")
if __name__=="__main__": main()
