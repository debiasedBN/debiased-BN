import os
import gc
import numpy as np
from config import SAVE_PATH, INPUT_PATH_PARQUET, METADATA_PATH, N_SPLITS, CLASSES, AGE_GROUPS, PHASES
from utils import create_new_modelpath
from data_processing import load_metadata, split_data, normalize_data
from training import train_the_model, final_results
import torch

def main():
    new_save_path = create_new_modelpath(SAVE_PATH)
    all_files_list = os.listdir(INPUT_PATH_PARQUET)
    metadata_df = load_metadata(METADATA_PATH, all_files_list)
    pooling_types = ['mean_probs', 'max']
    folds_scores = {
        phase: {
            'macro_group_avg': {pooling_type: [] for pooling_type in pooling_types},
            'macro_group_f1': {pooling_type: [] for pooling_type in pooling_types},
            **{
                age_group: {
                    label: {pooling_type: [] for pooling_type in pooling_types}
                    for label in CLASSES
                } for age_group in AGE_GROUPS
            }
        } for phase in PHASES
    }
    for fold_num in range(N_SPLITS):
        print(f"Processing fold {fold_num + 1}")
        fold_ids, fold_session_paths = split_data(metadata_df, all_files_list, fold_num, N_SPLITS)
        fold_dfs, global_mean, global_std = normalize_data(fold_session_paths, INPUT_PATH_PARQUET)
        for phase in PHASES:
            for age_group in fold_dfs[phase]:
                for label in fold_dfs[phase][age_group]:
                    print(f"{phase} {age_group} {label}: {len(fold_dfs[phase][age_group][label])} sessions")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        bestval_model = train_the_model(fold_dfs, new_save_path, fold_num, device)
        folds_scores = final_results(folds_scores, new_save_path, bestval_model, fold_num, fold_dfs, pooling_types, device)
        gc.collect()
    results_summary_path = os.path.join(new_save_path, 'summary_results.txt')
    with open(results_summary_path, 'w') as file:
        file.write("Final Summary Results\n")
        for phase in PHASES:
            for pooling_type in pooling_types:
                macro_group_avg = np.mean(folds_scores[phase]['macro_group_avg'][pooling_type]) * 100
                macro_group_std = np.std(folds_scores[phase]['macro_group_avg'][pooling_type]) * 100
                file.write(f"Phase:{phase.upper()}, Pooling:{pooling_type.upper()}, Macro-Averaged Accuracy: {macro_group_avg:.2f}%, Std: {macro_group_std:.2f}%\n")
                macro_group_f1 = np.mean(folds_scores[phase]['macro_group_f1'][pooling_type]) * 100
                macro_f1_std = np.std(folds_scores[phase]['macro_group_f1'][pooling_type]) * 100
                file.write(f"Phase:{phase.upper()}, Pooling:{pooling_type.upper()}, Macro-F1 Score: {macro_group_f1:.2f}%, Std: {macro_f1_std:.2f}%\n")
                for age_group in folds_scores[phase]:
                    if age_group in ['macro_group_avg', 'macro_group_f1']:
                        continue
                    for label in folds_scores[phase][age_group]:
                        mean_accuracy = np.mean(folds_scores[phase][age_group][label][pooling_type]) * 100
                        std_accuracy = np.std(folds_scores[phase][age_group][label][pooling_type]) * 100
                        file.write(f"AgeGroup:{age_group}, Class:{label.upper()}, Accuracy: {mean_accuracy:.2f}%, Std: {std_accuracy:.2f}%\n")
    print("Processing complete. Results saved in", new_save_path)

if __name__ == "__main__":
    main()
