import argparse
import os

import numpy as np
import torch
import torch.nn as nn

import train


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--output_root", type=str, default="./results")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = train.Config(data_root=args.data_root)
    cfg.base_output_dir = args.output_root
    if args.exp_name is not None:
        cfg.exp_name = args.exp_name
    if args.folds is not None:
        cfg.k_folds = args.folds
    cfg.evaluate_test = True
    cfg.reuse_existing_fold_models = True
    cfg.skip_finished_runs = False
    cfg.validate()
    train.configure_process_runtime()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    test_samples_by_subset = {
        subset: train.load_subset_samples(cfg, subset, "test")
        for subset in cfg.subsets
    }
    model_dir = cfg.get_output_dir()
    model_paths = [
        os.path.join(model_dir, f"model_fold_{fold_id}_best.pt")
        for fold_id in range(1, cfg.k_folds + 1)
    ]
    missing = [path for path in model_paths if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError("Missing checkpoint files:\n" + "\n".join(missing))
    models = []
    for path in model_paths:
        model = train.Hyper2OMModel(cfg).to(device)
        state = torch.load(path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        models.append(model)
    criterion = nn.BCEWithLogitsLoss().to(device)
    metrics = []
    prediction_rows = []
    for subset, samples in test_samples_by_subset.items():
        loader = train.build_loader(samples, cfg, shuffle=False, is_eval=True, desc=f"Test {subset}")
        fold_probabilities = []
        labels = None
        for model in models:
            _, probabilities, current_labels = train.run_one_epoch(
                model,
                loader,
                None,
                criterion,
                device,
                False,
                cfg=cfg,
                desc=f"Test {subset}",
            )
            fold_probabilities.append(probabilities)
            if labels is None:
                labels = current_labels
        probabilities = np.mean(fold_probabilities, axis=0)
        subset_metrics = train.calculate_metrics(labels, probabilities, threshold=0.5)
        train.append_result_to_csv(cfg, subset, subset_metrics, 0.5)
        train.append_detailed_record(cfg, subset, "Ensemble", "Test", subset_metrics)
        metrics.append(subset_metrics)
        for sample, probability in zip(samples, probabilities):
            prediction_rows.append(
                {
                    "subset": sample[2],
                    "sequence": sample[0],
                    "label": int(sample[1]),
                    "probability": float(probability),
                }
            )
        print(
            f"{subset}: AUC={subset_metrics['auc']:.6f} "
            f"AUPRC={subset_metrics['auprc']:.6f} "
            f"MCC={subset_metrics['mcc']:.6f}"
        )
    if metrics:
        mean_metrics = train.mean_metrics(metrics)
        train.append_result_to_csv(cfg, "AUCG_Mean", mean_metrics, 0.5)
        train.append_detailed_record(cfg, "AUCG_Mean", "Ensemble", "Test_Mean", mean_metrics)
        print(
            f"AUCG_Mean: AUC={mean_metrics['auc']:.6f} "
            f"AUPRC={mean_metrics['auprc']:.6f} "
            f"MCC={mean_metrics['mcc']:.6f}"
        )
    output_path = os.path.join(cfg.get_experiment_dir(), "results", "test_predictions.csv")
    train.write_rows(output_path, prediction_rows)
    train.write_sorted_results(cfg)


if __name__ == "__main__":
    main()
