import os
import json
from os import makedirs
from os.path import exists

import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_average_losses(losses_list):

    keys_list = list(losses_list[0].keys())
    num_items = len(losses_list)
    avg_metric_dict = {}

    # initialize average metrict dict wih zero
    for key in keys_list:
        avg_metric_dict[key] = 0

    # sum loss values
    for logs_dict in losses_list:
        for key, val in logs_dict.items():
            avg_metric_dict[key] += val

    # average values for number of items
    for key, value in avg_metric_dict.items():
        avg_metric_dict[key] = value/num_items

    return avg_metric_dict

def get_accuracy(logits, targets):

    if len(logits.shape) > 1:
        probabilities = torch.softmax(logits, dim=1)

        # Get the predicted class labels
        predicted_labels = torch.argmax(probabilities, dim=1)
        correct_predictions = (predicted_labels == targets).sum().item()
    else:
        correct_predictions = (logits == targets).sum().item()

    total_samples = targets.size(0)
    accuracy = correct_predictions / total_samples

    return accuracy

# returns logs specific to a stage during training steps
def get_logs(step_losses):

    logs = dict()

    logs["recon_loss"] = step_losses[0]
    logs["kl_loss"] = step_losses[1]
    logs["ce_loss"] = step_losses[2]
    logs["clst_loss"] = step_losses[3]
    logs["sep_loss"] = step_losses[4]
    logs["l1_loss"] = step_losses[5]
    logs["loss"] = step_losses[6]
    logs["total_loss"] = step_losses[7]
    logs["acc"] = step_losses[8]
    return logs

def create_dir(directory, verbose=False):
    if not exists(directory):
        makedirs(directory, exist_ok=True)
    if verbose:
        print(f"directory created: {directory}")


def load_parameters(json_file):
    try:
        with open(json_file, 'r') as file:
            parameters = json.load(file)
        return parameters
    except FileNotFoundError:
        print(f"Error: JSON file '{json_file}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: JSON file '{json_file}' is not a valid JSON file.")
        return None


def save_dict_as_json(dict_obj, path):
    if not isinstance(path, str):
        raise TypeError(f"path must be a string, got {type(path).__name__}")

    os.makedirs(path, exist_ok=True)
    filename = os.path.join(path, "hparams.json")
    with open(filename, 'w') as f:
        json.dump(dict_obj, f, indent=4)

    print(f"Dictionary saved as a json file at {filename}.\n")




