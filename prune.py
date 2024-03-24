
import argparse
import torch
import numpy as np

def bool_to_01(matrix):
    return np.where(matrix, 1, 0)

def pearson_corrcoef(x, y):
    mean_x, mean_y = torch.mean(x), torch.mean(y)
    cov_xy = torch.mean((x - mean_x) * (y - mean_y))
    std_x, std_y = torch.std(x), torch.std(y)
    corr_coef = cov_xy / (std_x * std_y)
    return corr_coef

parser = argparse.ArgumentParser()
parser.add_argument('--percent', type=float, default=0.9, help='pruning threshold (default: 0.9)')
args = parser.parse_args()


def pruner(model, threshold=0.9):
    model.cpu()
    valid_cols = []
    with torch.no_grad():
        for k, v in model.named_parameters():
            if len(v.shape) == 2:  # Consider only linear layers
                neurons = v.view(v.size(0), -1)
                correlations = []
                for i in range(neurons.size(1)):
                    for j in range(i+1, neurons.size(1)):
                        correlation = pearson_corrcoef(neurons[:, i], neurons[:, j]).item()
                        correlations.append(correlation)
                correlations = np.array(correlations)
                mask = (correlations >= threshold)
                pruned_neurons = neurons * mask.float().unsqueeze(0)
                v.data = pruned_neurons.view(v.size())
    return model
