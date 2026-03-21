"""
Pure Python inference pipeline for DCRNet.

Reconstructs MRI images from undersampled k-space data (accelerated MRI).
Input: .mat file prepared by Save_Input_Data_For_DCRNet.m
Output: reconstructed real/imaginary NIfTI and magnitude NIfTI
"""

import os
import sys
import tempfile

import numpy as np
import nibabel as nib
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_SINGLE_CH_DIR = os.path.join(_HERE, "PythonCodes", "Evaluation", "single_channel")
_MODEL_DIR = os.path.join(_SINGLE_CH_DIR, "Model")
CHECKPOINTS_DIR = os.path.join(_HERE, "checkpoints")

for _d in [_SINGLE_CH_DIR, _MODEL_DIR]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from DCRNet import DCRNet  # noqa: E402


_model_cache: dict = {}


def get_model(checkpoint_path: str, device: torch.device):
    """Load (or return cached) DCRNet model."""
    key = f"{checkpoint_path}::{device}"
    if key in _model_cache:
        return _model_cache[key]

    net = DCRNet(5)
    net = nn.DataParallel(net)
    net.load_state_dict(torch.load(checkpoint_path, map_location=device))
    net = net.to(device).eval()
    _model_cache[key] = net
    return net


def run_dcrnet(
    mat_path: str,
    *,
    checkpoint_path: str | None = None,
    output_dir: str | None = None,
    progress_fn=None,
) -> tuple[str, str, str]:
    """
    Run DCRNet MRI reconstruction in pure Python.

    Parameters
    ----------
    mat_path : str           – .mat file from Save_Input_Data_For_DCRNet.m
                               (must contain keys: mask, inputs_img, inputs_k)
    checkpoint_path : str    – path to DCRNet_single_channel.pth
                               (defaults to checkpoints/ subfolder)
    output_dir : str         – output directory (temp dir if None)

    Returns
    -------
    (real_path, imag_path, mag_path) – reconstructed NIfTI files
    """
    try:
        import mat73
    except ImportError:
        raise ImportError(
            "mat73 is required to load DCRNet input files. "
            "Install with: pip install mat73"
        )

    def _log(frac, msg):
        print(f"[{frac:.0%}] {msg}")
        if progress_fn:
            progress_fn(frac, msg)

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="dcrnet_")
    os.makedirs(output_dir, exist_ok=True)

    if checkpoint_path is None:
        checkpoint_path = os.path.join(CHECKPOINTS_DIR, "DCRNet_single_channel.pth")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _log(0.0, f"Device: {device}")

    _log(0.10, "Loading model …")
    net = get_model(checkpoint_path, device)

    _log(0.20, "Loading k-space data …")
    mat = mat73.loadmat(mat_path)

    mask = torch.from_numpy(np.array(mat["mask"])).float()
    mask = mask.unsqueeze(0).unsqueeze(0).to(device)

    image = np.array(mat["inputs_img"])
    image_r = torch.from_numpy(np.real(image)).float()
    image_i = torch.from_numpy(np.imag(image)).float()

    k0 = np.array(mat["inputs_k"])
    k0_r = torch.from_numpy(np.real(k0)).float()
    k0_i = torch.from_numpy(np.imag(k0)).float()

    n_slices = image_r.shape[2]
    recons_r = torch.zeros_like(image_r)
    recons_i = torch.zeros_like(image_i)

    _log(0.30, f"Reconstructing {n_slices} slices …")
    with torch.inference_mode():
        for j in range(n_slices):
            inp_r = image_r[:, :, j].unsqueeze(0).unsqueeze(0).to(device)
            inp_i = image_i[:, :, j].unsqueeze(0).unsqueeze(0).to(device)
            inp_kr = k0_r[:, :, j].unsqueeze(0).unsqueeze(0).to(device)
            inp_ki = k0_i[:, :, j].unsqueeze(0).unsqueeze(0).to(device)

            _, _, pred_r, pred_i = net(inp_r, inp_i, inp_kr, inp_ki, mask)

            recons_r[:, :, j] = pred_r.squeeze(0).squeeze(0)
            recons_i[:, :, j] = pred_i.squeeze(0).squeeze(0)

            if (j + 1) % max(1, n_slices // 10) == 0:
                _log(0.30 + 0.55 * (j + 1) / n_slices, f"  Slice {j + 1}/{n_slices}")

    recons_r_np = recons_r.numpy().astype(np.float32)
    recons_i_np = recons_i.numpy().astype(np.float32)
    magnitude = np.sqrt(recons_r_np**2 + recons_i_np**2)

    _log(0.90, "Saving …")
    affine = np.eye(4)
    real_path = os.path.join(output_dir, "DCRNet_real.nii.gz")
    imag_path = os.path.join(output_dir, "DCRNet_imag.nii.gz")
    mag_path  = os.path.join(output_dir, "DCRNet_magnitude.nii.gz")
    nib.save(nib.Nifti1Image(recons_r_np, affine), real_path)
    nib.save(nib.Nifti1Image(recons_i_np, affine), imag_path)
    nib.save(nib.Nifti1Image(magnitude, affine),    mag_path)

    _log(1.0, f"Done! Saved to {output_dir}")
    return real_path, imag_path, mag_path
