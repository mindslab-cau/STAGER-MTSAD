import torch
import numpy as np
import os
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error
from triad_tsad.evaluation.eval_utils import evaluate_w_threshold, evaluate_ensemble_best_modified

def run_thresholding(test_losses, train_losses, test_labels, threshold_methods, threshold_config, threshold_dir):
    """
    Run configured thresholding methods and compute evaluation metrics.
    
    Args:
        test_losses: Per-feature test scores.
        train_losses: Per-feature training scores. Retained for API compatibility.
        test_labels: Ground-truth test labels.
        threshold_methods: Thresholding method or methods to run, currently ["bestf1"].
        threshold_config: Thresholding configuration.
        threshold_dir: Directory for thresholding outputs.
        
    Returns:
        tuple: (evaluation_results, threshold_dict, predictions_dict)
            - evaluation_results: Metrics for each method.
            - threshold_dict: Thresholds for each method.
            - predictions_dict: Predictions for each method.
    """
    # Store evaluation metrics for each method.
    evaluation_results = {}
    
    # Store thresholds and predictions for each method.
    threshold_dict = {}
    predictions_dict = {}
    
    # Configure the output directory.
    output_dir = threshold_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert a string method to a list.
    if isinstance(threshold_methods, str):
        threshold_methods = [threshold_methods]
    
    if not threshold_methods:
        return evaluation_results, threshold_dict, predictions_dict
    
    print(f"\n[INFO] Thresholding methods: {', '.join(threshold_methods)}")
    
    # Accept a scalar FPR threshold or a list.
    fpr_threshold = threshold_config.get("fpr_threshold", 0.001)
    if not isinstance(fpr_threshold, list):
        fpr_threshold = [fpr_threshold]
    
    # Process each requested thresholding method.
    for method in threshold_methods:
        result_key = f"{method}_result"
        
        
        if method == "bestf1":
            print("[INFO] Running the Best F1 ensemble...")
            
            # Configure bestf1 outputs.
            fig_dir = os.path.join(output_dir, "bestf1_figures")
            os.makedirs(fig_dir, exist_ok=True)
            
            # Convert test_losses to a list of per-feature scores.
            predictions = [test_losses[i] for i in range(test_losses.shape[0])]
            
            # Evaluate every configured FPR threshold.
            for fpr in fpr_threshold:
                # Run evaluate_ensemble_best_modified.
                bestf1_result, bestf1_threshold_dict = evaluate_ensemble_best_modified(
                    attack_label=test_labels,
                    predictions=predictions,
                    fpr_threshold=fpr,
                    fig_dir=fig_dir,
                    is_brief=False
                )
                print(f"[INFO] BestF1 evaluation result (FPR {fpr}):", bestf1_result)
                
                # Include the FPR value in the result key.
                fpr_result_key = f"{result_key}_fpr{fpr}"
                evaluation_results[fpr_result_key] = bestf1_result
                
                # Store thresholds and predictions with the FPR value in their keys.
                threshold_dict[f"{method}_fpr{fpr}"] = bestf1_threshold_dict["thresholds"]
                predictions_dict[f"{method}_fpr{fpr}"] = bestf1_threshold_dict["combined_predictions"]
                
                # Save thresholds and predictions with the FPR value in their file names.
                np.save(os.path.join(threshold_dir, f"{method}_fpr{fpr}_threshold.npy"), bestf1_threshold_dict["thresholds"])
                np.save(os.path.join(threshold_dir, f"{method}_fpr{fpr}_prediction.npy"), bestf1_threshold_dict["combined_predictions"])
        
        else:
            print(f"[WARNING] Unknown thresholding method: {method}")
    
    # Convert NumPy arrays and scalars to JSON-serializable values.
    json_serializable_results = {}
    for key, value in evaluation_results.items():
        json_serializable_results[key] = {k: float(v) if isinstance(v, (np.ndarray, np.number)) else v for k, v in value.items()}
    
    # Save evaluation metrics as JSON.
    with open(os.path.join(output_dir, "evaluation_results.json"), 'w') as f:
        json.dump(json_serializable_results, f, indent=4)
    
    return evaluation_results, threshold_dict, predictions_dict

def save_evaluation_summary(evaluation_results, test_labels, threshold_dict, predictions_dict, eval_result_dir):
    """
    Save a summary of evaluation results.
    
    Args:
        evaluation_results: Evaluation results dictionary.
        test_labels: Ground-truth test labels.
        threshold_dict: Threshold dictionary.
        predictions_dict: Predictions dictionary.
        eval_result_dir: Directory for evaluation outputs.
    """
    # Save evaluation metrics as JSON.
    if evaluation_results:
        # Convert NumPy arrays and scalars to JSON-serializable values.
        json_serializable_results = {}
        for key, value in evaluation_results.items():
            json_serializable_results[key] = {k: float(v) if isinstance(v, (np.ndarray, np.number)) else v for k, v in value.items()}
        
        with open(os.path.join(eval_result_dir, "evaluation_results.json"), 'w') as f:
            json.dump(json_serializable_results, f, indent=4)
    
    # Save test labels.
    if test_labels is not None:
        np.save(os.path.join(eval_result_dir, "test_labels.npy"), test_labels)
