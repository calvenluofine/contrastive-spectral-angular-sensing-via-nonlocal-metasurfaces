import torch
import matplotlib.pyplot as plt
import numpy as np
import math
from torch.optim import SGD, Adagrad, Adadelta, RMSprop, Adam, Adamax
from torch.optim import AdamW, LBFGS
import torch.nn.functional as F
import torch.nn as nn




def get_optimizer(optimizer_name, model_params, learning_rate, **kwargs):

    optimizer_name = optimizer_name.strip().lower()


    weight_decay = kwargs.get('weight_decay', 0)

    if optimizer_name == "sgd":
        return SGD(
            model_params,
            lr=learning_rate,
            momentum=kwargs.get('momentum', 0),
            dampening=kwargs.get('dampening', 0),
            weight_decay=weight_decay,
            nesterov=kwargs.get('nesterov', False)
        )
    elif optimizer_name == "sgd+momentum":
        return SGD(
            model_params,
            lr=learning_rate,
            momentum=kwargs.get('momentum', 0.9),
            dampening=kwargs.get('dampening', 0),
            weight_decay=weight_decay,
            nesterov=kwargs.get('nesterov', False)
        )
    elif optimizer_name == "adagrad":
        return Adagrad(
            model_params,
            lr=learning_rate,
            lr_decay=kwargs.get('lr_decay', 0),
            weight_decay=weight_decay,
            eps=kwargs.get('eps', 1e-10)
        )
    elif optimizer_name == "adadelta":
        return Adadelta(
            model_params,
            lr=learning_rate,
            rho=kwargs.get('rho', 0.9),
            eps=kwargs.get('eps', 1e-6),
            weight_decay=weight_decay
        )
    elif optimizer_name == "rmsprop":
        return RMSprop(
            model_params,
            lr=learning_rate,
            alpha=kwargs.get('alpha', 0.99),
            eps=kwargs.get('eps', 1e-8),
            weight_decay=weight_decay,
            momentum=kwargs.get('momentum', 0),
            centered=kwargs.get('centered', False)
        )
    elif optimizer_name == "adam":
        return Adam(
            model_params,
            lr=learning_rate,
            betas=kwargs.get('betas', (0.9, 0.999)),
            eps=kwargs.get('eps', 1e-8),
            weight_decay=weight_decay,
            amsgrad=kwargs.get('amsgrad', False)
        )
    elif optimizer_name == "adamax":
        return Adamax(
            model_params,
            lr=learning_rate,
            betas=kwargs.get('betas', (0.9, 0.999)),
            eps=kwargs.get('eps', 1e-8),
            weight_decay=weight_decay
        )
    elif optimizer_name == "adamw":
        return AdamW(
            model_params,
            lr=learning_rate,
            betas=kwargs.get('betas', (0.9, 0.999)),
            eps=kwargs.get('eps', 1e-8),
            weight_decay=kwargs.get('weight_decay', 0.01),
            amsgrad=kwargs.get('amsgrad', False)
        )
    elif optimizer_name == "l-bfgs":
        return LBFGS(
            model_params,
            lr=learning_rate,
            max_iter=kwargs.get('max_iter', 20),
            max_eval=kwargs.get('max_eval', None),
            tolerance_grad=kwargs.get('tolerance_grad', 1e-7),
            tolerance_change=kwargs.get('tolerance_change', 1e-9),
            history_size=kwargs.get('history_size', 100),
            line_search_fn=kwargs.get('line_search_fn', None)
        )
    else:
        raise ValueError(f"no-support: {optimizer_name}")





def generate_mask_matrix(batch_size, device ,num=31):


    total_size = batch_size * num * num
    mask = torch.zeros((total_size, total_size), dtype=torch.int, device=device)


    for b in range(batch_size):
        batch_start_row = b * num * num
        batch_start_col = b * num * num


        diag_block_idx = 0
        for i in range(num):
            for j in range(num):

                block_start_row = batch_start_row + i * num
                block_start_col = batch_start_col + j * num


                if i == j:


                    pos = diag_block_idx % num


                    mask[block_start_row + pos, block_start_col + pos] = 1


                    diag_block_idx += 1

    return mask


def spectrum_fingerprint(spectra):
    import hashlib
    values = spectra.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str((values.shape, values.dtype.str)).encode())
    digest.update(values.tobytes())
    return digest.hexdigest()
