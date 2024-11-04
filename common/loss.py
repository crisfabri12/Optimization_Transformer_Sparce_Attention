import torch
import numpy as np


def mpjpe(output_3D, out_target):
	loss = torch.mean(torch.norm(output_3D - out_target, dim=-1))

	return loss


def weighted_mpjpe(predicted, target, w_mpjpe):
    # Calcular la diferencia y la norma
    diff = predicted - target  # [B, F, 17, 3]
    norm = torch.norm(diff, dim=3)  # [B, F, 17]

    # Ajustar la forma de w_mpjpe para que coincida con 'norm'
    if w_mpjpe.dim() == 1 and w_mpjpe.size(0) == norm.size(2):
        w_mpjpe = w_mpjpe.view(1, 1, -1).expand_as(norm)  # [B, F, 17]
    else:
        raise ValueError(f"Forma inesperada de 'w_mpjpe': {w_mpjpe.shape}")

    # Calcular la pérdida ponderada
    loss = torch.mean(w_mpjpe * norm)

    return loss

def temporal_consistency(predicted, target, w_mpjpe):
    dif_seq = predicted[:,1:,:,:] - predicted[:,:-1,:,:]
    weights_joints = torch.ones_like(dif_seq).cuda()
    weights_joints = torch.mul(weights_joints.permute(0,1,3,2), w_mpjpe).permute(0,1,3,2)
    dif_seq = torch.mean(torch.multiply(weights_joints, torch.square(dif_seq)))

    return dif_seq


def mean_velocity(predicted, target, axis=0):
    velocity_predicted = predicted[:,1:] - predicted[:,:-1]
    velocity_target = target[:,1:] - target[:,:-1]

    return torch.mean(torch.norm(velocity_predicted - velocity_target, dim=len(target.shape)-1))





