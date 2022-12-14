import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


def euler2mat(angle, scaling=False, translation=False):
    """Convert euler angles to rotation matrix.
     Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
    Args:
        angle: rotation angle along 3 axis (a, b, y) in radians -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
    """
    B = angle.size(0)
    euler_angle = 'bay'

    if euler_angle == 'bay':
        a, b, y = angle[:, 0], angle[:, 1], angle[:, 2]
        x, y, z = b, a, y
    elif euler_angle == 'aby':
        x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]
    else:
        raise NotImplementedError

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach() * 0
    ones = zeros.detach() + 1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz, cosz, zeros,
                        zeros, zeros, ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros, siny,
                        zeros, ones, zeros,
                        -siny, zeros, cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros, cosx, -sinx,
                        zeros, sinx, cosx], dim=1).reshape(B, 3, 3)

    rotMat = xmat @ ymat @ zmat
    # rotMat = zmat
    # rotMat = ymat @ zmat
    # rotMat = xmat @ ymat
    # rotMat = xmat @ zmat

    if scaling:
        v_scale = angle[:,3]
        v_trans = angle[:,4:]
    else:
        v_trans = angle[:,3:]

    if scaling:
        # one = torch.ones_like(v_scale).detach()
        # t_scale = torch.stack([v_scale, one, one,
        #                        one, v_scale, one,
        #                        one, one, v_scale], dim=1).view(B, 3, 3)
        rotMat = rotMat * v_scale.unsqueeze(1).unsqueeze(1)

    if translation:
        rotMat = torch.cat([rotMat, v_trans.view([B, 3, 1]).cuda()], 2)  # F.affine_grid takes 3x4
    else:
        rotMat = torch.cat([rotMat, torch.zeros([B, 3, 1]).cuda().detach()], 2)  # F.affine_grid takes 3x4

    return rotMat


def quat2mat(quat, scaling=False, translation=False):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: first three coeff of quaternion of rotation. fourht is then computed to have a norm of 1 -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = torch.cat([quat[:,:1].detach()*0 + 1, quat], dim=1)
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    rotMat = torch.cat([rotMat, torch.zeros([B, 3, 1]).cuda().detach()], 2)  # F.affine_grid takes 3x4

    return rotMat

