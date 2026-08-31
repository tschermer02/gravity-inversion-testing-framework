"""Run and compare the controlled four-run E09B lambda sweep."""
from __future__ import annotations
import argparse,csv,json,subprocess,sys
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
from cnn_inversion_3d import failure_mode_analysis as analysis

RUNS={
 "E09B-1":{"slug":"E09B_1_depth1p5","lambda_depth":1.5,"lambda_sensitivity":1.0},
 "E09B-2":{"slug":"E09B_2_depth2p0","lambda_depth":2.0,"lambda_sensitivity":1.0},
 "E09B-3":{"slug":"E09B_3_depth3p0","lambda_depth":3.0,"lambda_sensitivity":1.0},
 "E09B-4":{"slug":"E09B_4_depth2p0_sens0p5","lambda_depth":2.0,"lambda_sensitivity":0.5},
}
BASELINE={"slug":"E09B_integrated_sensitivity","lambda_depth":1.0,"lambda_sensitivity":1.0}

def _run(command:list[str])->None:
    print(" ".join(command),flush=True); subprocess.run(command,check=True)
def _json(path:Path)->Any: return json.loads(path.read_text(encoding="utf-8"))
def _valid_training(path:Path,cfg:dict[str,Any])->bool:
    meta=path/"training_metadata.json"
    if not meta.exists() or not (path/"best_model.keras").exists(): return False
    data=_json(meta); tc=data["training_configuration"]; loss=data["loss"]["e09b"]
    return tc["architecture"]=="single_plane_asymmetric_2d_unet_sensitivity_loss" and abs(loss["lambda_depth"]-cfg["lambda_depth"])<1e-12 and abs(loss["lambda_sensitivity"]-cfg["lambda_sensitivity"])<1e-12 and tc["random_seed"]==20260727
def _valid_prediction(path:Path,cfg:dict[str,Any])->bool:
    marker=path/"sweep_configuration.json"; metrics=path/"prediction_metrics.json"
    return marker.exists() and metrics.exists() and _json(marker)==cfg and len(_json(metrics))==100
def _valid_analysis(path:Path)->bool: return (path/"gravity_consistency_metrics.csv").exists()
def _write_csv(path:Path,rows:list[dict[str,Any]])->None: analysis.write_csv(path,rows)

def compare(root:Path,dataset:Path,output:Path)->None:
    experiments={"E09B":str(root/"prediction_outputs"/BASELINE["slug"])}
    experiments.update({label:str(root/"prediction_outputs"/cfg["slug"]) for label,cfg in RUNS.items()})
    analysis.EXPERIMENTS=experiments; sample_rows,by=analysis.collect(root,dataset); aggregates=analysis.aggregate(by)
    configs={"E09B":BASELINE,**RUNS}; overall=[]; training=[]
    for row in aggregates:
        label=row["experiment"]; cfg=configs[label]; tdir=root/"training_outputs"/cfg["slug"]
        history=list(csv.DictReader((tdir/"training_history.csv").open(encoding="utf-8"))); best=min(history,key=lambda x:float(x["val_loss"])); final=history[-1]
        tr={"experiment":label,"best_epoch":int(best["epoch"]),"final_epoch":int(final["epoch"]),"best_validation_loss":float(best["val_loss"])}
        for key in ("density_loss","depth_profile_loss","z_center_loss","depth_loss","sensitivity_loss"):
            tr[f"val_{key}_at_best_epoch"]=float(best[f"val_{key}"])
            tr[f"val_{key}_improved_after_best"]=min(float(x[f"val_{key}"]) for x in history[int(best["epoch"]):]) < float(best[f"val_{key}"])
        training.append(tr)
        overall.append({"experiment":label,"lambda_density":1.0,"lambda_depth":cfg["lambda_depth"],"alpha_center":1.0,"lambda_sensitivity":cfg["lambda_sensitivity"],
          "mean_mse":row["mean_mse"],"mean_iou":row["mean_iou"],"mean_dice":row["mean_dice"],
          "mean_absolute_top_depth_error_m":row["mean_absolute_top_depth_error_m"],"mean_absolute_bottom_depth_error_m":row["mean_absolute_bottom_depth_error_m"],
          "mean_absolute_center_depth_error_m":row["mean_absolute_z_center_error_m"],"mean_absolute_thickness_error_m":row["mean_absolute_thickness_error_m"],
          "gravity_mse":row["mean_gravity_mse"],"gravity_rmse":row["mean_gravity_rmse"],"gravity_mae":row["mean_gravity_mae"],
          "gravity_relative_l2":row["mean_gravity_relative_l2"],"gravity_correlation":row["mean_gravity_correlation"],**tr})
    base={r["sample_id"]:r for r in by["E09B"]}; paired=[]
    metrics=("iou","dice","mse","absolute_top_depth_error_m","absolute_bottom_depth_error_m","absolute_z_center_error_m","absolute_thickness_error_m","gravity_rmse","gravity_correlation")
    higher={"iou","dice","gravity_correlation"}
    for label in RUNS:
        current={r["sample_id"]:r for r in by[label]}
        for metric in metrics:
            diffs=np.array([current[s][metric]-base[s][metric] for s in base],float); improved=diffs>0 if metric in higher else diffs<0
            paired.append({"experiment":label,"baseline":"E09B","metric":metric,"mean_paired_difference":np.nanmean(diffs),"median_paired_difference":np.nanmedian(diffs),"samples_improved":int(np.sum(improved)),"samples_worsened":int(np.sum(~improved)),"percent_improved":100*np.mean(improved)})
    output.mkdir(parents=True,exist_ok=True); _write_csv(output/"overall_comparison.csv",overall); _write_csv(output/"sample_metrics.csv",sample_rows); _write_csv(output/"paired_metrics.csv",paired); _write_csv(output/"training_summary.csv",training)
    labels=[r["experiment"] for r in overall]; plot_metrics=(("mean_iou","IoU ↑"),("mean_dice","Dice ↑"),("mean_absolute_top_depth_error_m","Top MAE ↓"),("mean_absolute_bottom_depth_error_m","Bottom MAE ↓"),("mean_absolute_thickness_error_m","Thickness MAE ↓"),("mean_mse","MSE ↓"),("gravity_rmse","Gravity RMSE ↓"),("gravity_correlation","Gravity corr ↑"))
    fig,axes=plt.subplots(2,4,figsize=(16,8),constrained_layout=True)
    for ax,(key,title) in zip(axes.flat,plot_metrics): ax.bar(labels,[r[key] for r in overall]); ax.set_title(title); ax.tick_params(axis="x",rotation=45); ax.grid(axis="y",alpha=.2)
    fig.savefig(output/"e09b_lambda_sweep_metrics.png",dpi=180); plt.close(fig)
    fig,axes=plt.subplots(1,5,figsize=(18,4),constrained_layout=True)
    for label,cfg in RUNS.items():
        hist=list(csv.DictReader((root/"training_outputs"/cfg["slug"] /"training_history.csv").open()))
        for ax,key in zip(axes,("val_loss","val_density_loss","val_depth_profile_loss","val_z_center_loss","val_sensitivity_loss")): ax.plot([float(r[key]) for r in hist],label=label); ax.set_title(key); ax.grid(alpha=.2)
    axes[-1].legend(); fig.savefig(output/"training_loss_comparison.png",dpi=180); plt.close(fig)
    winners={title:min(overall,key=lambda r:r[key])["experiment"] if "↑" not in title else max(overall,key=lambda r:r[key])["experiment"] for key,title in plot_metrics}
    (output/"README.md").write_text("# E09B Lambda Sweep\n\nControlled sweep of lambda_depth and lambda_sensitivity; architecture and all other settings are fixed.\n\n## Metric leaders\n"+"\n".join(f"- {k}: {v}" for k,v in winners.items())+"\n\nInspect overall and paired tables for Pareto tradeoffs; no winner is selected from validation loss alone.\n",encoding="utf-8")

def main()->None:
    p=argparse.ArgumentParser(); p.add_argument("--dataset",type=Path,default=Path("datasets/canonical_single_plane_train2000")); p.add_argument("--resume",action="store_true"); p.add_argument("--overwrite",action="store_true"); p.add_argument("--gravity-scale",type=float,default=.22938017547130585); a=p.parse_args(); root=Path.cwd(); dataset=a.dataset.resolve()
    for i,(label,cfg) in enumerate(RUNS.items(),1):
        out=root/"training_outputs"/cfg["slug"]; print(f"[TRAIN {i}/4] {label}",flush=True)
        if a.resume and _valid_training(out,cfg): print("  verified; skipping"); continue
        cmd=[sys.executable,"-m","cnn_inversion_3d.train","--dataset",str(dataset),"--output",str(out),"--architecture","single_plane_asymmetric_2d_unet_sensitivity_loss","--e09a-lambda-density","1","--e09a-lambda-depth",str(cfg["lambda_depth"]),"--e09a-alpha-center","1","--e09b-lambda-sensitivity",str(cfg["lambda_sensitivity"]),"--e09b-sensitivity-gamma","0.5","--e09b-weight-min","0.5","--e09b-weight-max","5","--base-filters","8","--gravity-scale-summary",str(dataset/"training_distribution.json"),"--gravity-scale-method","percentile_99","--learning-rate","0.001","--batch-size","2","--epochs","100","--seed","20260727"]
        if a.overwrite: cmd.append("--overwrite")
        _run(cmd); assert _valid_training(out,cfg),f"Invalid training output for {label}"
    for i,(label,cfg) in enumerate(RUNS.items(),1):
        out=root/"prediction_outputs"/cfg["slug"]; print(f"[PREDICT {i}/4] {label}",flush=True)
        if a.resume and _valid_prediction(out,cfg): print("  verified; skipping"); continue
        cmd=[sys.executable,"-m","cnn_inversion_3d.predict","--dataset",str(dataset),"--model",str(root/"training_outputs"/cfg["slug"]/"best_model.keras"),"--manifest","test_manifest.csv","--output",str(out),"--samples","100","--threshold","0.1","--gravity-scale",str(a.gravity_scale)]
        if a.overwrite: cmd.append("--overwrite")
        _run(cmd); (out/"sweep_configuration.json").write_text(json.dumps(cfg,indent=2)); assert _valid_prediction(out,cfg)
    for i,(label,cfg) in enumerate(RUNS.items(),1):
        out=root/"prediction_outputs"/cfg["slug"]; print(f"[ANALYZE {i}/4] {label}",flush=True)
        if a.resume and _valid_analysis(out): print("  verified; skipping"); continue
        _run([sys.executable,"-m","cnn_inversion_3d.analyze_predictions","--dataset",str(dataset),"--predictions",str(out),"--metrics","prediction_metrics.json","--evaluate-gravity-consistency"]); assert _valid_analysis(out)
    print("[COMPARE] Original E09B + E09B-1/2/3/4"); compare(root,dataset,root/"analysis_outputs"/"E09B_lambda_sweep")
if __name__=="__main__": main()
