import os
import time
import copy
import gc
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import f1_score

from config import BATCH_SIZE, RANDOM_SEED, PHASES, CLASSES, NUM_CLASSES, APPLY_GROUP_DRO, STEP_SIZE, EMA_ALPHA, PATIENCE, EPOCHS, EPOCH_STEPS, LEARNING_RATE, WEIGHT_DECAY, DEBIASED_BN, DFR, CHANNEL_NUM, RAW_SEQUENCE_LENGTH
from model import define_model

def return_accuracies(outputs, labels):
    if labels.ndim > 1 and labels.size(1) == NUM_CLASSES:
        labels = torch.argmax(labels, dim=1)
    _, predicted = torch.max(outputs, dim=1)
    return (predicted == labels).float()

def group_loss_accuracy(losses, accuracies, labels, ageGroups, groups):
    group_losses = torch.zeros(len(groups)).to(losses.device)
    group_accuracies = torch.zeros(len(groups)).to(losses.device)
    group_counts = torch.zeros(len(groups)).to(losses.device)
    if labels.ndim > 1 and labels.size(1) == NUM_CLASSES:
        labels = labels.argmax(dim=1)
    for group_index, (age_group, label) in enumerate(groups):
        mask = (labels == CLASSES.index(label)) & (ageGroups == [g for g, _ in groups].index(age_group))
        group_counts[group_index] = mask.sum()
        group_losses[group_index] = losses[mask].mean() if mask.any() else 0
        group_accuracies[group_index] = accuracies[mask].mean() if mask.any() else 0
    return group_losses, group_accuracies, group_counts, group_losses.max(), group_accuracies.min()

def group_dro_loss(group_losses, adv_probs, group_counts, step_size=STEP_SIZE):
    normalized_group_losses = group_losses / (group_counts + 1e-6)
    adv_probs = adv_probs * torch.exp(step_size * normalized_group_losses).clone().detach()
    adv_probs = adv_probs / adv_probs.sum()
    robust_loss = torch.dot(normalized_group_losses, adv_probs)
    return robust_loss, adv_probs

def train_the_model(fold_dfs, new_save_path, fold_num, device):
    """
    Train the EEG model.

    Args:
        fold_dfs: DataFrames dictionary for current fold.
        new_save_path: Directory to save models.
        fold_num: Current fold number.
        device: Computation device.

    Returns:
        best_model: Model with best validation performance.
    """
    from data_processing import batch_generator, eval_generator
    groups = [(age_group, label) for age_group in sorted(fold_dfs["train"].keys()) for label in CLASSES]
    train_generator = batch_generator(fold_dfs["train"], device, training=True)
    valid_generator = batch_generator(fold_dfs["valid"], device)
    model, optimizer = define_model()
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    val_loss_ema = 1.0
    val_accuracy_ema = 0.0
    best_val_accuracy = -float('inf')
    patience_counter = 0
    best_model_path = os.path.join(new_save_path, f"best_val_model_{fold_num}.pt")
    if APPLY_GROUP_DRO:
        group_counts = torch.zeros(len(groups)).to(device)
        adv_probs = (torch.ones(len(groups)).to(device) / len(groups)).clone().detach()
    train_losses, val_losses, train_accuracies, val_accuracies, val_loss_ema_list, val_accuracy_ema_list = [], [], [], [], [], []
    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        model.train()
        running_loss, running_accuracy = 0.0, 0.0
        for _ in range(EPOCH_STEPS):
            inputs, labels, ageGroups = next(train_generator)
            optimizer.zero_grad()
            outputs = model(inputs)
            losses = criterion(outputs, labels)
            accuracies = return_accuracies(outputs, labels)
            group_losses, group_accuracies, group_counts, worst_group_loss, worst_group_accuracy = group_loss_accuracy(losses, accuracies, labels, ageGroups, groups)
            if APPLY_GROUP_DRO:
                loss, adv_probs = group_dro_loss(group_losses, adv_probs, group_counts)
            else:
                loss = losses.mean()
            loss.backward()
            optimizer.step()
            running_loss += worst_group_loss.item() if worst_group_loss is not None else loss.item()
            running_accuracy += worst_group_accuracy.item() if worst_group_accuracy is not None else group_accuracies.mean().item()
        model.eval()
        val_loss, val_accuracy = 0.0, 0.0
        with torch.no_grad():
            for _ in range(EPOCH_STEPS):
                inputs, labels, ageGroups = next(valid_generator)
                outputs = model(inputs)
                losses = criterion(outputs, labels)
                accuracies = return_accuracies(outputs, labels)
                group_losses, group_accuracies, group_counts, worst_group_loss, worst_group_accuracy = group_loss_accuracy(losses, accuracies, labels, ageGroups, groups)
                val_loss += worst_group_loss.item() if worst_group_loss is not None else losses.mean().item()
                val_accuracy += worst_group_accuracy.item() if worst_group_accuracy is not None else group_accuracies.mean().item()
        epoch_loss = running_loss / EPOCH_STEPS
        epoch_accuracy = running_accuracy / EPOCH_STEPS
        val_epoch_loss = val_loss / EPOCH_STEPS
        val_epoch_accuracy = val_accuracy / EPOCH_STEPS
        val_loss_ema = EMA_ALPHA * val_epoch_loss + (1 - EMA_ALPHA) * val_loss_ema
        val_accuracy_ema = EMA_ALPHA * val_epoch_accuracy + (1 - EMA_ALPHA) * val_accuracy_ema
        epoch_duration = time.time() - epoch_start_time
        minutes, seconds = divmod(epoch_duration, 60)
        print(f'Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f}, Acc: {epoch_accuracy:.4f}, Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_accuracy:.4f}, Val Loss EMA: {val_loss_ema:.4f}, Val Acc EMA: {val_accuracy_ema:.4f} - Time: {int(minutes):02d}:{int(seconds):02d}')
        train_losses.append(epoch_loss)
        val_losses.append(val_epoch_loss)
        train_accuracies.append(epoch_accuracy)
        val_accuracies.append(val_epoch_accuracy)
        val_loss_ema_list.append(val_loss_ema)
        val_accuracy_ema_list.append(val_accuracy_ema)
        if val_accuracy_ema > best_val_accuracy:
            print("Validation accuracy EMA increased, saving model...")
            best_val_accuracy = val_accuracy_ema
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping...")
            break
    if DFR:
        best_model_state = copy.deepcopy(model.state_dict())
        torch.save(best_model_state, best_model_path)
    # Plot metrics
    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(20, 6))
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs_range, val_losses, 'r--', label='Validation Loss')
    plt.title('Loss Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, train_accuracies, 'b-', label='Training Accuracy')
    plt.plot(epochs_range, val_accuracies, 'r--', label='Validation Accuracy')
    plt.title('Accuracy Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, val_loss_ema_list, 'b-', label='Validation Loss EMA')
    plt.plot(epochs_range, val_accuracy_ema_list, 'r--', label='Validation Accuracy EMA')
    plt.title('EMA Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('EMA Value')
    plt.legend()
    plt_path = os.path.join(new_save_path, f"metrics_{fold_num}.png")
    plt.savefig(plt_path)
    plt.show()
    best_model_state = torch.load(best_model_path)
    best_model, _ = define_model()
    best_model.load_state_dict(best_model_state)
    best_model = best_model.to(device)
    best_model.eval()
    return best_model

def final_results(folds_scores, new_save_path, bestval_model, fold_num, fold_dfs, pooling_types, device):
    """
    Evaluate model on test data and save results.

    Args:
        folds_scores: Dictionary for storing scores.
        new_save_path: Directory for saving results.
        bestval_model: Best validation model.
        fold_num: Current fold number.
        fold_dfs: DataFrames dictionary.
        pooling_types: List of pooling types.
        device: Computation device.

    Returns:
        Updated folds_scores.
    """
    from data_processing import eval_generator
    for phase in PHASES:
        result_path = os.path.join(new_save_path, f'results_{phase}.txt')
        with open(result_path, 'a') as file:
            print(f"\nFold:{fold_num+1}")
            file.write(f"\nFold:{fold_num}\n")
            for pooling_type in pooling_types:
                print(f"\nPhase:{phase}, Pooling Type:{pooling_type}")
                file.write(f"\nPhase:{phase}, Pooling Type:{pooling_type}\n")
                group_accuracies = []
                all_true_labels = []
                all_pred_labels = []
                groups = [(age_group, label) for age_group in sorted(fold_dfs[phase].keys()) for label in CLASSES]
                for age_group, label in groups:
                    label_index = CLASSES.index(label)
                    true_labels = []
                    pred_labels = []
                    for df in fold_dfs[phase][age_group][label]:
                        batch_data = eval_generator(df, device, RAW_SEQUENCE_LENGTH)
                        with torch.no_grad():
                            logits = bestval_model(batch_data)
                            logits = logits.view(-1, NUM_CLASSES)
                            predictions = F.softmax(logits, dim=-1)
                            if pooling_type == 'mean_probs':
                                class_label = predictions.mean(dim=0).argmax(dim=-1).cpu().numpy()
                                pred_labels.append(class_label)
                                true_labels.append(label_index)
                            elif pooling_type == 'max':
                                class_label = predictions.max(dim=0).values.argmax(dim=-1).cpu().numpy()
                                pred_labels.append(class_label)
                                true_labels.append(label_index)
                    correct_predictions = (np.array(pred_labels) == np.array(true_labels)).sum()
                    accuracy = correct_predictions / len(true_labels)
                    folds_scores[phase][age_group][label][pooling_type].append(accuracy)
                    group_accuracies.append(accuracy)
                    all_true_labels.extend(true_labels)
                    all_pred_labels.extend(pred_labels)
                    print(f"Accuracy:{accuracy:.4f}, AgeGroup:{age_group}, Class:{label}")
                    file.write(f"Accuracy:{accuracy:.4f}, AgeGroup:{age_group}, Class:{label}\n")
                macro_avg_accuracy = np.mean(group_accuracies)
                folds_scores[phase]['macro_group_avg'][pooling_type].append(macro_avg_accuracy)
                print(f"Macro-Averaged Accuracy:{macro_avg_accuracy:.4f}")
                file.write(f"Macro-Averaged Accuracy:{macro_avg_accuracy:.4f}\n")
                macro_f1 = f1_score(all_true_labels, all_pred_labels, average='macro')
                folds_scores[phase]['macro_group_f1'][pooling_type].append(macro_f1)
                print(f"Macro-Averaged F1 Score:{macro_f1:.4f}")
                file.write(f"Macro-Averaged F1 Score:{macro_f1:.4f}\n")
    return folds_scores
