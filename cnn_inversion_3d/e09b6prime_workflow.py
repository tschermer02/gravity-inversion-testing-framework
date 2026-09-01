"""Generate, train, predict, analyze, and compare controlled E09B-6-prime."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from cnn_inversion_3d import failure_mode_analysis as analysis
from cnn_inversion_3d.gravity_consistency import build_cnn_forward_model_context

DATASET_NAME = "canonical_single_plane_train10000_balanced_size_density"
TRAINING_NAME = "E09B_6prime"
BASELINE_NAME = "E09B_6_amplitude"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--resume", action="store_true")
    result.add_argument("--overwrite", action="store_true")
    result.add_argument("--seed", type=int, default=20260901)
    return result


def run(command: list[str]) -> None:
    print(" ".join(command), flush=True); subprocess.run(command, check=True)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream: return list(csv.DictReader(stream))


def manifests_match(left: Path, right: Path) -> bool:
    return read_csv(left) == read_csv(right)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def test_sample_files_match(dataset: Path, reference: Path) -> bool:
    rows = read_csv(reference/"test_manifest.csv")
    return all(file_sha256(dataset/row["relative_path"]) ==
               file_sha256(reference/row["relative_path"]) for row in rows)


def valid_dataset(dataset: Path, reference: Path) -> bool:
    required = ("metadata.json", "train_manifest.csv", "validation_manifest.csv",
                "test_manifest.csv", "training_distribution.json")
    if not all((dataset/name).exists() for name in required): return False
    metadata = read_json(dataset/"metadata.json")
    return (metadata.get("split_counts") == {"train":10000,"validation":1000,"test":100}
            and metadata.get("parent_reference_experiment") == "E09B-6"
            and manifests_match(dataset/"test_manifest.csv", reference/"test_manifest.csv")
            and test_sample_files_match(dataset, reference))


def valid_training(path: Path, dataset: Path) -> bool:
    if not (path/"best_model.keras").exists() or not (path/"training_metadata.json").exists(): return False
    data = read_json(path/"training_metadata.json"); loss = data.get("loss",{}).get("e09b") or {}
    controlled = data.get("controlled_experiment") or {}
    return (data.get("trainable_parameters") == 190592
            and Path(data["training_configuration"]["dataset_directory"]).resolve() == dataset.resolve()
            and loss.get("lambda_depth") == 2.0 and loss.get("lambda_amplitude") == 1.0
            and loss.get("lambda_sensitivity") == 1.0 and not loss.get("small_body_weighting")
            and controlled.get("parent_reference_experiment") == "E09B-6")


def prediction_control(dataset: Path, model: Path) -> dict[str, Any]:
    return {"experiment":"E09B-6'","dataset":str(dataset.resolve()),
            "model":str(model.resolve()),
            "test_manifest_sha256":file_sha256(dataset/"test_manifest.csv"),"sample_count":100}


def valid_prediction(path: Path, dataset: Path, model: Path) -> bool:
    marker=path/"e09b6prime_prediction_control.json";metrics=path/"prediction_metrics.json"
    return (marker.exists() and metrics.exists() and len(read_json(metrics)) == 100
            and read_json(marker) == prediction_control(dataset,model))


def add_training_control_metadata(path: Path) -> None:
    metadata_path = path/"training_metadata.json"; data = read_json(metadata_path)
    data["controlled_experiment"] = {
        "display_name":"E09B-6'", "filesystem_name":TRAINING_NAME,
        "parent_reference_experiment":"E09B-6",
        "scientific_variable":"training dataset size and balanced size-density coverage",
        "unchanged_model_and_loss":True, "small_body_loss_weighting":False,
        "forward_gravity_training_loss":False,
    }
    metadata_path.write_text(json.dumps(data,indent=2),encoding="utf-8")


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.nanmean([float(row[key]) for row in rows]))


def compare(root: Path, dataset: Path, baseline_dataset: Path, output: Path) -> None:
    if not manifests_match(dataset/"test_manifest.csv", baseline_dataset/"test_manifest.csv"):
        raise RuntimeError("E09B-6 and E09B-6-prime test manifests do not match exactly.")
    analysis.EXPERIMENTS = {
        "E09B-6": str(root/"prediction_outputs"/BASELINE_NAME),
        "E09B-6'": str(root/"prediction_outputs"/TRAINING_NAME),
    }
    sample_rows, by = analysis.collect(root, dataset)
    ids = [row["sample_id"] for row in by["E09B-6"]]
    ranked = sorted(ids, key=lambda sid: (next(r for r in by["E09B-6"] if r["sample_id"]==sid)["true_occupied_cells"],sid))
    groups = {sid:("small" if i<25 else "large" if i>=75 else "medium") for i,sid in enumerate(ranked)}
    prediction_dirs = {"E09B-6":root/"prediction_outputs"/BASELINE_NAME,
                       "E09B-6'":root/"prediction_outputs"/TRAINING_NAME}
    context = build_cnn_forward_model_context(dataset/"metadata.json")
    for row in sample_rows:
        true_density=float(row["true_body_mean_density"]); predicted=float(row["predicted_mean_density_in_true_body"])
        prediction_file=prediction_dirs[row["experiment"]]/row["prediction_path"]
        with np.load(prediction_file) as saved:
            predicted_volume=np.asarray(saved["predicted_density"],dtype=float).squeeze()
        with np.load(dataset/row["sample_path"]) as saved:
            true_gravity=np.asarray(saved["gravity"],dtype=float)
        recovered_gravity=np.asarray(context.forward_model.calculate(predicted_volume),dtype=float).squeeze()
        true_peak=float(np.max(true_gravity)); predicted_peak=float(np.max(recovered_gravity))
        row.update({"body_size_group":groups[row["sample_id"]],"true_density_contrast":true_density,
            "predicted_density_contrast":predicted,"density_error":predicted-true_density,
            "absolute_density_error":abs(predicted-true_density),
            "relative_density_error":abs(predicted-true_density)/max(abs(true_density),1e-8),
            "density_ratio":predicted/max(true_density,1e-8),"true_gravity_peak":true_peak,
            "predicted_gravity_peak":predicted_peak,"gravity_peak_ratio":predicted_peak/max(true_peak,1e-8),
            "true_body_volume":row["true_occupied_cells"],"predicted_body_volume":row["predicted_occupied_cells"]})
    by={label:[row for row in sample_rows if row["experiment"]==label] for label in analysis.EXPERIMENTS}
    metric_specs=(("mean_iou","iou"),("mean_dice","dice"),("density_mse","mse"),
        ("top_depth_mae_m","absolute_top_depth_error_m"),("bottom_depth_mae_m","absolute_bottom_depth_error_m"),
        ("center_depth_mae_m","absolute_z_center_error_m"),("thickness_mae_m","absolute_thickness_error_m"),
        ("density_contrast_mae","absolute_density_error"),("relative_density_error","relative_density_error"),
        ("gravity_mse","gravity_mse"),("gravity_rmse","gravity_rmse"),("gravity_mae","gravity_mae"),
        ("gravity_relative_l2","gravity_relative_l2"),("gravity_correlation","gravity_correlation"),
        ("gravity_peak_ratio","gravity_peak_ratio"),("volume_ratio","volume_ratio"),("density_ratio","density_ratio"))
    overall=[]
    for label,rows in by.items():
        density_errors=np.asarray([r["density_error"] for r in rows],float)
        overall.append({"experiment":label,**{name:mean(rows,key) for name,key in metric_specs},
            "density_contrast_rmse":float(np.sqrt(np.mean(np.square(density_errors)))),
            "density_bias":float(np.mean(density_errors))})
    maps={row["experiment"]:row for row in overall}; paired=[]
    for name,key in metric_specs:
        old=maps["E09B-6"][name]; new=maps["E09B-6'"][name]
        paired.append({"metric":name,"e09b6":old,"e09b6prime":new,"absolute_difference":new-old,
                       "percent_change_relative_to_e09b6":100*(new-old)/old if old else float("nan")})
    for name in ("density_contrast_rmse","density_bias"):
        old=maps["E09B-6"][name];new=maps["E09B-6'"][name]
        paired.append({"metric":name,"e09b6":old,"e09b6prime":new,"absolute_difference":new-old,
                       "percent_change_relative_to_e09b6":100*(new-old)/abs(old) if old else float("nan")})
    per_sample=[]; row_maps={label:{r["sample_id"]:r for r in rows} for label,rows in by.items()}
    for sid in ids:
        row={"sample_id":sid,"body_size_group":groups[sid]}
        for label in by:
            for _,key in metric_specs: row[f"{label}_{key}"]=row_maps[label][sid][key]
        per_sample.append(row)
    grouped=[]
    group_keys=("iou","dice","absolute_top_depth_error_m","absolute_bottom_depth_error_m",
                "absolute_thickness_error_m","absolute_density_error","relative_density_error",
                "gravity_rmse","gravity_relative_l2","gravity_correlation")
    for label,rows in by.items():
        for group in ("small","medium","large"):
            selected=[r for r in rows if r["body_size_group"]==group]
            grouped.append({"experiment":label,"body_size_group":group,"sample_count":len(selected),
                            **{f"mean_{key}":mean(selected,key) for key in group_keys}})
    calibration=[]
    for label,rows in by.items():
        for group in ("all","small","medium","large"):
            selected=rows if group=="all" else [r for r in rows if r["body_size_group"]==group]
            true=np.asarray([r["true_density_contrast"] for r in selected]);pred=np.asarray([r["predicted_density_contrast"] for r in selected])
            slope,intercept=np.polyfit(true,pred,1)
            calibration.append({"experiment":label,"body_size_group":group,"sample_count":len(selected),
                "slope":float(slope),"intercept":float(intercept),"density_bias":float(np.mean(pred-true)),
                "density_mae":float(np.mean(np.abs(pred-true))),"density_rmse":float(np.sqrt(np.mean(np.square(pred-true)))),
                "density_correlation":float(np.corrcoef(true,pred)[0,1])})
    output.mkdir(parents=True,exist_ok=True)
    for filename,rows in (("overall_comparison.csv",overall),("aggregate_changes.csv",paired),
                          ("paired_sample_metrics.csv",per_sample),("body_size_group_metrics.csv",grouped),
                          ("density_calibration.csv",calibration),("gravity_amplitude_diagnostics.csv",sample_rows)):
        analysis.write_csv(output/filename,rows)
    fig,axes=plt.subplots(2,2,figsize=(10,9),constrained_layout=True)
    for axis,group in zip(axes.flat,("all","small","medium","large")):
        for label,rows in by.items():
            selected=rows if group=="all" else [r for r in rows if r["body_size_group"]==group]
            axis.scatter([r["true_density_contrast"] for r in selected],[r["predicted_density_contrast"] for r in selected],s=14,alpha=.5,label=label)
        axis.plot([.2,1],[.2,1],"k--",linewidth=1);axis.set(xlabel="True density",ylabel="Predicted density",title=group.title());axis.legend()
    fig.savefig(output/"true_vs_predicted_density.png",dpi=180);plt.close(fig)
    for filename,xkey in (("density_ratio_vs_gravity_peak_ratio.png","density_ratio"),
                          ("volume_ratio_vs_gravity_peak_ratio.png","volume_ratio"),
                          ("body_volume_vs_gravity_peak_ratio.png","true_body_volume")):
        fig,axis=plt.subplots(figsize=(7,5))
        for label,rows in by.items():axis.scatter([r[xkey] for r in rows],[r["gravity_peak_ratio"] for r in rows],s=15,alpha=.5,label=label)
        axis.axhline(1,color="black",linestyle="--");axis.set(xlabel=xkey,ylabel="Gravity peak ratio");axis.grid(alpha=.2);axis.legend();fig.tight_layout();fig.savefig(output/filename,dpi=180);plt.close(fig)
    old,new=maps["E09B-6"],maps["E09B-6'"]
    groupmap={(r["experiment"],r["body_size_group"]):r for r in grouped}
    oldgap=groupmap[("E09B-6","large")]["mean_dice"]-groupmap[("E09B-6","small")]["mean_dice"]
    newgap=groupmap[("E09B-6'","large")]["mean_dice"]-groupmap[("E09B-6'","small")]["mean_dice"]
    density_corr={label:float(np.corrcoef([r["density_ratio"] for r in rows],[r["gravity_peak_ratio"] for r in rows])[0,1]) for label,rows in by.items()}
    volume_corr={label:float(np.corrcoef([r["volume_ratio"] for r in rows],[r["gravity_peak_ratio"] for r in rows])[0,1]) for label,rows in by.items()}
    recommendation=("forward-model consistency" if new["gravity_rmse"]>=old["gravity_rmse"] else
                    "density-loss redesign" if abs(new["density_bias"])>=abs(old["density_bias"]) else "architecture")
    (output/"README.md").write_text(f"""# E09B-6 vs E09B-6-prime

The models and losses are identical; only training dataset size/coverage changes.

1. Overall IoU/Dice changes: {new['mean_iou']-old['mean_iou']:+.5f} / {new['mean_dice']-old['mean_dice']:+.5f}.
2. Small-body IoU/Dice changes: {groupmap[("E09B-6'","small")]['mean_iou']-groupmap[("E09B-6","small")]['mean_iou']:+.5f} / {groupmap[("E09B-6'","small")]['mean_dice']-groupmap[("E09B-6","small")]['mean_dice']:+.5f}.
3. Large-minus-small Dice gap changed from {oldgap:.5f} to {newgap:.5f}.
4. Top/bottom/thickness MAE changes: {new['top_depth_mae_m']-old['top_depth_mae_m']:+.3f}, {new['bottom_depth_mae_m']-old['bottom_depth_mae_m']:+.3f}, {new['thickness_mae_m']-old['thickness_mae_m']:+.3f} m.
5. Density MAE/bias changes: {new['density_contrast_mae']-old['density_contrast_mae']:+.6f} / {new['density_bias']-old['density_bias']:+.6f} g/cm³.
6. Gravity RMSE change: {new['gravity_rmse']-old['gravity_rmse']:+.6g} mGal.
7. Mean gravity peak ratio changed from {old['gravity_peak_ratio']:.4f} to {new['gravity_peak_ratio']:.4f}; movement toward 1 indicates reduced amplitude bias.
8. Density-ratio vs peak-ratio correlations: {density_corr}; volume-ratio correlations: {volume_corr}. Larger absolute correlation is the stronger association, not causal proof.
9. The paired and grouped changes show whether E09B-6 was data-limited; improvement must occur overall and especially for small bodies without degrading other objectives.
10. Evidence-based next focus under the simple decision rule used here: **{recommendation}**. No new loss or architecture is implemented here.
""",encoding="utf-8")


def main() -> None:
    args=parser().parse_args();root=Path.cwd();reference=root/"datasets"/"canonical_single_plane_train2000";dataset=root/"datasets"/DATASET_NAME
    review=root/"analysis_outputs"/"E09B_6prime_dataset_review"
    print("[GENERATE / VALIDATE DATASET]",flush=True)
    if not (args.resume and valid_dataset(dataset,reference) and (review/"dataset_summary.csv").exists()):
        command=[sys.executable,"-m","dataset_generation.generate_e09b6prime_dataset","--output",str(dataset),"--reference-dataset",str(reference),"--seed",str(args.seed)]
        if args.overwrite:command.append("--overwrite")
        run(command)
    if not valid_dataset(dataset,reference):raise RuntimeError("Generated E09B-6-prime dataset failed validation.")
    training=root/"training_outputs"/TRAINING_NAME
    print("[TRAIN E09B-6']",flush=True)
    if not (args.resume and valid_training(training,dataset)):
        command=[sys.executable,"-m","cnn_inversion_3d.train","--dataset",str(dataset),"--output",str(training),
            "--architecture","single_plane_asymmetric_2d_unet_sensitivity_loss","--e09a-lambda-density","1",
            "--e09a-lambda-depth","2","--e09a-alpha-center","1","--e09b-lambda-sensitivity","1",
            "--e09b-lambda-amplitude","1","--e09b-sensitivity-gamma","0.5","--e09b-weight-min","0.5",
            "--e09b-weight-max","5","--base-filters","8","--gravity-scale-summary",str(dataset/"training_distribution.json"),
            "--gravity-scale-method","percentile_99","--learning-rate","0.001","--batch-size","2","--epochs","100","--seed","20260727"]
        if args.overwrite:command.append("--overwrite")
        run(command);add_training_control_metadata(training)
    if not valid_training(training,dataset):raise RuntimeError("E09B-6-prime training output is mismatched or incomplete.")
    prediction=root/"prediction_outputs"/TRAINING_NAME
    print("[PREDICT E09B-6']",flush=True)
    model_path=training/"best_model.keras"
    if not (args.resume and valid_prediction(prediction,dataset,model_path)):
        scale=read_json(dataset/"training_distribution.json")["absolute_percentile_99"]
        command=[sys.executable,"-m","cnn_inversion_3d.predict","--dataset",str(dataset),"--model",str(model_path),
                 "--manifest","test_manifest.csv","--output",str(prediction),"--samples","100","--threshold","0.1","--gravity-scale",str(scale)]
        if args.overwrite:command.append("--overwrite")
        run(command)
        (prediction/"e09b6prime_prediction_control.json").write_text(
            json.dumps(prediction_control(dataset,model_path),indent=2),encoding="utf-8")
    if not valid_prediction(prediction,dataset,model_path):raise RuntimeError("E09B-6-prime predictions are incomplete or mismatched.")
    print("[ANALYZE E09B-6']",flush=True)
    if not (args.resume and (prediction/"gravity_consistency_metrics.csv").exists()):
        run([sys.executable,"-m","cnn_inversion_3d.analyze_predictions","--dataset",str(dataset),"--predictions",str(prediction),
             "--metrics","prediction_metrics.json","--evaluate-gravity-consistency"])
    if not (prediction/"gravity_consistency_metrics.csv").exists():raise RuntimeError("Gravity analysis is incomplete.")
    print("[COMPARE E09B-6 vs E09B-6']",flush=True)
    compare(root,dataset,reference,root/"analysis_outputs"/"E09B6_vs_E09B6prime")


if __name__=="__main__":main()
