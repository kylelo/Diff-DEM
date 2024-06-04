from typing import Tuple
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy

def mae(input, target):
    with torch.no_grad():
        loss = nn.L1Loss()
        output = loss(input, target)
    return output

def get_valid_img_pair(
    target: torch.tensor, 
    input: torch.tensor,
    max_thres: float = 80,
    scale_factor: float = 80,
    valid_thres: float = 0.01,
    invert: bool = False
) -> Tuple[torch.tensor, torch.tensor, float]:
    """Keep valid pixel for input and target and return number of valid pixels."""

    input = (input + 1) / 2.0
    target = (target + 1) / 2.0

    mask = target > valid_thres

    input *= scale_factor
    target *= scale_factor

    input[input > max_thres] = max_thres
    target[target > max_thres] = max_thres

    input = input * 1e3
    target = target * 1e3

    # print(torch.max(input).item(), torch.min(input).item())
    # print(torch.max(target).item(), torch.min(target).item())

    return target[mask], input[mask], mask.sum()

def masked_mae(target, input, valid_thres=0.0001):
    trg, src = target.clone(), input.clone()
    trg, src, n_valid = get_valid_img_pair(target, input, valid_thres=valid_thres)

    diff_abs = torch.abs(trg - src)

    mae = diff_abs.sum() / (n_valid + 1e-8)

    return mae

def masked_rmse(target, input, valid_thres=0.0001):
    trg, src = target.clone(), input.clone()
    trg, src, n_valid = get_valid_img_pair(target, input, valid_thres=valid_thres)

    diff_sqr = torch.pow(trg - src, 2)

    rmse = diff_sqr.sum() / (n_valid + 1e-8)
    rmse = torch.sqrt(rmse)

    return rmse


def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)