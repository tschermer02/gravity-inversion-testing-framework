"""Sequential E09B-9/10/11 training and controlled four-model analysis."""
from __future__ import annotations
import argparse,csv,json,subprocess,sys
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
from cnn_inversion_3d import failure_mode_analysis as analysis

DATASET="canonical_single_plane_train10000_balanced_size_density"
RUNS={
 "E09B-9":{"slug":"E09B_9_body_density","body":1.0,"gravity":0.0},
 "E09B-10":{"slug":"E09B_10_gravity_physics","body":0.0,"gravity":.001},
 "E09B-11":{"slug":"E09B_11_body_density_gravity_physics","body":1.0,"gravity":.001},
}
ALL={"E09B-6'":{"slug":"E09B_6prime","body":0.0,"gravity":0.0},**RUNS}

def run(command:list[str])->None: print(" ".join(command),flush=True);subprocess.run(command,check=True)
def load(path:Path)->Any:return json.loads(path.read_text(encoding="utf-8"))
def write(path:Path,rows:list[dict[str,Any]])->None:analysis.write_csv(path,rows)
def mean(rows,key):return float(np.nanmean([float(row[key]) for row in rows]))
def valid_training(path:Path,cfg:dict[str,Any])->bool:
 m=path/"training_metadata.json"
 if not m.exists() or not (path/"best_model.keras").exists():return False
 d=load(m);loss=d.get("loss",{}).get("e09b") or {};control=d.get("controlled_experiment") or {}
 return d.get("trainable_parameters")==190592 and loss.get("lambda_depth")==2 and loss.get("lambda_amplitude")==1 and loss.get("lambda_body_density")==cfg["body"] and loss.get("lambda_gravity")==cfg["gravity"] and control.get("parent_experiment")=="E09B-6'"
def valid_prediction(path:Path,cfg:dict[str,Any])->bool:
 marker=path/"e09b_density_physics_configuration.json";metrics=path/"prediction_metrics.json"
 return marker.exists() and metrics.exists() and load(marker)==cfg and len(load(metrics))==100
def annotate(path:Path,label:str,cfg:dict[str,Any])->None:
 p=path/"training_metadata.json";d=load(p);d["controlled_experiment"]={"experiment_name":label,"parent_experiment":"E09B-6'","lambda_body_density":cfg["body"],"lambda_gravity":cfg["gravity"],"scientific_variables":["body-density loss","gravity physics loss"],"architecture_unchanged":True};p.write_text(json.dumps(d,indent=2),encoding="utf-8")

def compare(root:Path,dataset:Path,output:Path)->None:
 analysis.EXPERIMENTS={label:str(root/"prediction_outputs"/cfg["slug"]) for label,cfg in ALL.items()}
 sample_rows,by=analysis.collect(root,dataset);ids=[r["sample_id"] for r in by["E09B-6'"]]
 reference={r["sample_id"]:r for r in by["E09B-6'"]};ranked=sorted(ids,key=lambda sid:(reference[sid]["true_occupied_cells"],sid));groups={sid:"small" if i<25 else "large" if i>=75 else "medium" for i,sid in enumerate(ranked)}
 dirs={label:root/"prediction_outputs"/cfg["slug"] for label,cfg in ALL.items()}
 for row in sample_rows:
  with np.load(dirs[row["experiment"]]/row["prediction_path"]) as saved: truth=np.asarray(saved["true_density"]).squeeze();prediction=np.asarray(saved["predicted_density"]).squeeze()
  mask=truth>0;true_density=float(np.mean(truth[mask]));pred_density=float(np.mean(prediction[mask]));body_mse=float(np.mean(np.square(prediction[mask]-truth[mask])))
  gdir=dirs[row["experiment"]]/"gravity_consistency"/row["sample_id"]
  true_peak=float(np.max(np.load(gdir/"true_gravity.npy")));pred_peak=float(np.max(np.load(gdir/"recovered_gravity.npy")))
  row.update(body_size_group=groups[row["sample_id"]],true_density_contrast=true_density,predicted_density_contrast=pred_density,density_error=pred_density-true_density,absolute_density_error=abs(pred_density-true_density),relative_density_error=abs(pred_density-true_density)/max(true_density,1e-8),true_body_voxel_density_mse=body_mse,density_ratio=pred_density/max(true_density,1e-8),true_gravity_peak=true_peak,predicted_gravity_peak=pred_peak,gravity_peak_ratio=pred_peak/max(true_peak,1e-8))
 by={label:[r for r in sample_rows if r["experiment"]==label] for label in ALL}
 fields=("mse","iou","dice","absolute_top_depth_error_m","absolute_bottom_depth_error_m","absolute_z_center_error_m","absolute_thickness_error_m","absolute_density_error","relative_density_error","true_body_voxel_density_mse","gravity_mse","gravity_rmse","gravity_mae","gravity_relative_l2","gravity_correlation","gravity_peak_ratio","volume_ratio","density_ratio")
 overall=[];density_rows=[];gravity_rows=[]
 for label,rows in by.items():
  de=np.asarray([r["density_error"] for r in rows]);true=np.asarray([r["true_density_contrast"] for r in rows]);pred=np.asarray([r["predicted_density_contrast"] for r in rows])
  base={"experiment":label,**{f"mean_{key}":mean(rows,key) for key in fields},"density_contrast_rmse":float(np.sqrt(np.mean(de**2))),"density_bias":float(np.mean(de)),"density_correlation":float(np.corrcoef(true,pred)[0,1])};overall.append(base)
  density_rows.append({key:value for key,value in base.items() if key=="experiment" or "density" in key})
  gravity_rows.append({key:value for key,value in base.items() if key=="experiment" or "gravity" in key})
 group_rows=[]
 group_fields=("iou","dice","absolute_density_error","relative_density_error","absolute_top_depth_error_m","absolute_bottom_depth_error_m","absolute_thickness_error_m","gravity_rmse","gravity_relative_l2","gravity_correlation")
 for label,rows in by.items():
  for group in ("small","medium","large"):
   selected=[r for r in rows if r["body_size_group"]==group];group_rows.append({"experiment":label,"body_size_group":group,"sample_count":len(selected),**{f"mean_{key}":mean(selected,key) for key in group_fields}})
 paired=[];maps={label:{r["sample_id"]:r for r in rows} for label,rows in by.items()};higher={"iou","dice"}
 for label in RUNS:
  for key in ("iou","dice","absolute_density_error","true_body_voxel_density_mse","absolute_top_depth_error_m","absolute_bottom_depth_error_m","absolute_thickness_error_m","gravity_rmse","gravity_relative_l2"):
   differences=np.asarray([maps[label][sid][key]-maps["E09B-6'"][sid][key] for sid in ids]);improved=differences>0 if key in higher else differences<0
   paired.append({"comparison":f"E09B-6' -> {label}","metric":key,"mean_paired_difference":float(np.mean(differences)),"median_paired_difference":float(np.median(differences)),"samples_improved":int(np.sum(improved)),"samples_worsened":int(np.sum(~improved)),"percent_improved":float(100*np.mean(improved))})
 training=[]
 for label,cfg in ALL.items():
  history=list(csv.DictReader((root/"training_outputs"/cfg["slug"]/"training_history.csv").open(encoding="utf-8")));best=min(history,key=lambda r:float(r["val_loss"]));row={"experiment":label,"best_epoch":int(best["epoch"]),"best_validation_loss":float(best["val_loss"]),"final_epoch":int(history[-1]["epoch"]),"trainable_parameters":190592}
  for key,value in best.items():
   if "loss" in key:row[f"selected_checkpoint_{key}"]=float(value)
  training.append(row)
 output.mkdir(parents=True,exist_ok=True)
 for filename,rows in (("overall_comparison.csv",overall),("sample_metrics.csv",sample_rows),("paired_metrics.csv",paired),("body_size_group_metrics.csv",group_rows),("density_metrics.csv",density_rows),("gravity_metrics.csv",gravity_rows),("training_summary.csv",training)):write(output/filename,rows)
 labels=list(ALL)
 def bars(filename,keys,titles,source=overall):
  lookup={r["experiment"]:r for r in source};fig,axes=plt.subplots(1,len(keys),figsize=(5*len(keys),4),constrained_layout=True);axes=np.atleast_1d(axes)
  for ax,key,title in zip(axes,keys,titles):ax.bar(labels,[lookup[l][key] for l in labels]);ax.set_title(title);ax.tick_params(axis="x",rotation=35);ax.grid(axis="y",alpha=.2)
  fig.savefig(output/filename,dpi=180);plt.close(fig)
 bars("overall_metrics.png",("mean_iou","mean_dice","mean_mse"),("IoU","Dice","MSE"));bars("density_metrics.png",("mean_absolute_density_error","density_contrast_rmse","mean_true_body_voxel_density_mse"),("Density MAE","Density RMSE","Body voxel MSE"));bars("gravity_metrics.png",("mean_gravity_rmse","mean_gravity_relative_l2","mean_gravity_correlation"),("Gravity RMSE","Relative L2","Correlation"))
 small=[r for r in group_rows if r["body_size_group"]=="small"];bars("small_body_metrics.png",("mean_iou","mean_dice","mean_absolute_density_error","mean_gravity_rmse"),("Small IoU","Small Dice","Small density MAE","Small gravity RMSE"),small)
 fig,ax=plt.subplots(figsize=(7,5));
 for label,rows in by.items():ax.scatter([r["absolute_density_error"] for r in rows],[r["gravity_rmse"] for r in rows],s=12,alpha=.45,label=label)
 ax.set(xlabel="Density error",ylabel="Gravity RMSE");ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(output/"density_vs_gravity.png",dpi=180);plt.close(fig)
 for filename,xkey,ykey in (("true_density_vs_predicted_density.png","true_density_contrast","predicted_density_contrast"),("density_ratio_vs_gravity_peak_ratio.png","density_ratio","gravity_peak_ratio"),("volume_ratio_vs_gravity_peak_ratio.png","volume_ratio","gravity_peak_ratio"),("density_error_vs_gravity_rmse.png","absolute_density_error","gravity_rmse")):
  fig,ax=plt.subplots(figsize=(7,5));
  for label,rows in by.items():ax.scatter([r[xkey] for r in rows],[r[ykey] for r in rows],s=12,alpha=.45,label=label)
  if xkey=="true_density_contrast":ax.plot([.2,1],[.2,1],"k--")
  ax.set(xlabel=xkey,ylabel=ykey);ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(output/filename,dpi=180);plt.close(fig)
 fig,ax=plt.subplots(figsize=(8,5));x=np.arange(3);width=.2
 for i,label in enumerate(labels):lookup={(r["experiment"],r["body_size_group"]):r for r in group_rows};ax.bar(x+(i-1.5)*width,[lookup[(label,g)]["mean_absolute_density_error"] for g in ("small","medium","large")],width,label=label)
 ax.set_xticks(x,("small","medium","large"));ax.set_ylabel("Density MAE");ax.legend();fig.tight_layout();fig.savefig(output/"body_density_error_by_size.png",dpi=180);plt.close(fig)
 om={r["experiment"]:r for r in overall};leaders={"overlap":max(overall,key=lambda r:r["mean_dice"])["experiment"],"geometry":min(overall,key=lambda r:r["mean_absolute_top_depth_error_m"]+r["mean_absolute_bottom_depth_error_m"]+r["mean_absolute_thickness_error_m"])["experiment"],"density":min(overall,key=lambda r:r["mean_absolute_density_error"])["experiment"],"body voxel density":min(overall,key=lambda r:r["mean_true_body_voxel_density_mse"])["experiment"],"gravity":min(overall,key=lambda r:r["mean_gravity_rmse"])["experiment"],"small body":max(small,key=lambda r:r["mean_dice"])["experiment"]}
 rank={label:0 for label in labels}
 for key,higher in (("mean_dice",True),("mean_absolute_top_depth_error_m",False),("mean_absolute_density_error",False),("mean_true_body_voxel_density_mse",False),("mean_gravity_rmse",False)):
  for position,label in enumerate(sorted(labels,key=lambda x:om[x][key],reverse=higher),1):rank[label]+=position
 leaders["overall compromise"]=min(rank,key=rank.get)
 baseline=om["E09B-6'"]
 effects="\n".join(f"- {label}: ΔDice={om[label]['mean_dice']-baseline['mean_dice']:+.5f}, Δdensity MAE={om[label]['mean_absolute_density_error']-baseline['mean_absolute_density_error']:+.6f}, Δgravity RMSE={om[label]['mean_gravity_rmse']-baseline['mean_gravity_rmse']:+.6f}" for label in RUNS)
 gm={(r["experiment"],r["body_size_group"]):r for r in group_rows};base_label="E09B-6'";b=baseline;e9=om["E09B-9"];e10=om["E09B-10"];e11=om["E09B-11"]
 geometry=lambda r:r["mean_absolute_top_depth_error_m"]+r["mean_absolute_bottom_depth_error_m"]+r["mean_absolute_thickness_error_m"]
 closer=lambda new,old:abs(new-1)<abs(old-1)
 answers=f"""## Direct scientific answers

1. E09B-9 density MAE change: {e9['mean_absolute_density_error']-b['mean_absolute_density_error']:+.6f}.
2. E09B-9 absolute density-bias change: {abs(e9['density_bias'])-abs(b['density_bias']):+.6f}.
3. E09B-9 body-voxel MSE change: {e9['mean_true_body_voxel_density_mse']-b['mean_true_body_voxel_density_mse']:+.6f}.
4. E09B-9 combined top/bottom/thickness MAE change: {geometry(e9)-geometry(b):+.3f} m.
5. E09B-9 gravity RMSE change without physics supervision: {e9['mean_gravity_rmse']-b['mean_gravity_rmse']:+.6f} mGal.
6. E09B-9 small-body density MAE change: {gm[('E09B-9','small')]['mean_absolute_density_error']-gm[(base_label,'small')]['mean_absolute_density_error']:+.6f}.
7. E09B-10 gravity RMSE change: {e10['mean_gravity_rmse']-b['mean_gravity_rmse']:+.6f} mGal.
8. E09B-10 mean gravity peak ratio changed {b['mean_gravity_peak_ratio']:.4f} → {e10['mean_gravity_peak_ratio']:.4f}; closer to one: {closer(e10['mean_gravity_peak_ratio'],b['mean_gravity_peak_ratio'])}.
9. E09B-10 density MAE change: {e10['mean_absolute_density_error']-b['mean_absolute_density_error']:+.6f}.
10. E09B-10 IoU/Dice changes: {e10['mean_iou']-b['mean_iou']:+.5f} / {e10['mean_dice']-b['mean_dice']:+.5f}.
11. E09B-10 small-body Dice change: {gm[('E09B-10','small')]['mean_dice']-gm[(base_label,'small')]['mean_dice']:+.5f}.
12. E09B-10 volume/density ratios: {e10['mean_volume_ratio']:.4f} / {e10['mean_density_ratio']:.4f}; departures from one must be considered alongside gravity improvement.
13. E09B-11 changes versus baseline—body MSE {e11['mean_true_body_voxel_density_mse']-b['mean_true_body_voxel_density_mse']:+.6f}, gravity RMSE {e11['mean_gravity_rmse']-b['mean_gravity_rmse']:+.6f}.
14. E09B-11 Dice relative to E09B-9/E09B-10: {e11['mean_dice']-e9['mean_dice']:+.5f} / {e11['mean_dice']-e10['mean_dice']:+.5f}.
15. E09B-11 density MAE and gravity RMSE changes versus baseline: {e11['mean_absolute_density_error']-b['mean_absolute_density_error']:+.6f} / {e11['mean_gravity_rmse']-b['mean_gravity_rmse']:+.6f}.
16. E09B-11 combined vertical-geometry error change: {geometry(e11)-geometry(b):+.3f} m.
"""
 (output/"README.md").write_text("# E09B Density–Physics Ablation\n\nAll models use the unchanged E09B-6-prime architecture and identical dataset/test samples. Gravity inversion is non-unique; gravity agreement is not treated as proof of density accuracy.\n\n## Objective leaders\n"+"\n".join(f"- Best {k}: {v}" for k,v in leaders.items())+"\n\n## Controlled effects\n"+effects+"\n\n"+answers+"\nDo not select a model from validation loss alone; interpret density, geometry, gravity, and small-body performance separately.\n",encoding="utf-8")

def main()->None:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--resume",action="store_true");p.add_argument("--overwrite",action="store_true");a=p.parse_args();root=Path.cwd();dataset=root/"datasets"/DATASET;scale=load(dataset/"training_distribution.json")["absolute_percentile_99"]
 for i,(label,cfg) in enumerate(RUNS.items(),1):
  out=root/"training_outputs"/cfg["slug"];print(f"[TRAIN {i}/3] {label}",flush=True)
  if a.resume and valid_training(out,cfg):print("  verified; skipping");continue
  command=[sys.executable,"-m","cnn_inversion_3d.train","--dataset",str(dataset),"--output",str(out),"--architecture","single_plane_asymmetric_2d_unet_sensitivity_loss","--e09a-lambda-density","1","--e09a-lambda-depth","2","--e09a-alpha-center","1","--e09b-lambda-sensitivity","1","--e09b-lambda-amplitude","1","--e09b-lambda-body-density",str(cfg["body"]),"--e09b-lambda-gravity",str(cfg["gravity"]),"--e09b-sensitivity-gamma","0.5","--e09b-weight-min","0.5","--e09b-weight-max","5","--base-filters","8","--gravity-scale-summary",str(dataset/"training_distribution.json"),"--gravity-scale-method","percentile_99","--learning-rate","0.001","--batch-size","2","--epochs","100","--seed","20260727"]
  if a.overwrite:command.append("--overwrite")
  run(command);annotate(out,label,cfg)
  if not valid_training(out,cfg):raise RuntimeError(f"Invalid training output: {label}")
 for i,(label,cfg) in enumerate(RUNS.items(),1):
  out=root/"prediction_outputs"/cfg["slug"];print(f"[PREDICT {i}/3] {label}",flush=True)
  if a.resume and valid_prediction(out,cfg):print("  verified; skipping");continue
  command=[sys.executable,"-m","cnn_inversion_3d.predict","--dataset",str(dataset),"--model",str(root/"training_outputs"/cfg["slug"]/"best_model.keras"),"--manifest","test_manifest.csv","--output",str(out),"--samples","100","--threshold","0.1","--gravity-scale",str(scale)]
  if a.overwrite:command.append("--overwrite")
  run(command);(out/"e09b_density_physics_configuration.json").write_text(json.dumps(cfg,indent=2),encoding="utf-8")
  if not valid_prediction(out,cfg):raise RuntimeError(f"Invalid predictions: {label}")
 for i,(label,cfg) in enumerate(RUNS.items(),1):
  out=root/"prediction_outputs"/cfg["slug"];print(f"[ANALYZE {i}/3] {label}",flush=True);volumes=out/"gravity_consistency"/next(iter(load(out/"prediction_metrics.json")))["prediction_path"].replace("_prediction.npz","")/"true_gravity.npy"
  if not(a.resume and (out/"gravity_consistency_metrics.csv").exists() and volumes.exists()):run([sys.executable,"-m","cnn_inversion_3d.analyze_predictions","--dataset",str(dataset),"--predictions",str(out),"--metrics","prediction_metrics.json","--evaluate-gravity-consistency","--save-gravity-volumes"])
 # Ensure baseline volumes needed for amplitude comparisons, without retraining/predicting.
 base=root/"prediction_outputs"/"E09B_6prime"
 first=next(iter(load(base/"prediction_metrics.json")))["prediction_path"].replace("_prediction.npz","")
 if not(base/"gravity_consistency"/first/"true_gravity.npy").exists():run([sys.executable,"-m","cnn_inversion_3d.analyze_predictions","--dataset",str(dataset),"--predictions",str(base),"--metrics","prediction_metrics.json","--evaluate-gravity-consistency","--save-gravity-volumes"])
 print("[COMPARE] E09B-6' / E09B-9 / E09B-10 / E09B-11",flush=True);compare(root,dataset,root/"analysis_outputs"/"E09B_density_physics_ablation")
if __name__=="__main__":main()
