"""
DCRNet – Gradio Web Interface
=====================================
Web UI for DCRNet accelerated MRI reconstruction.

Launch:
    python app.py                   # CPU
    python app.py --share           # public Gradio link
    python app.py --server-port 8080

Docker:
    docker compose up               # see docker-compose.yml
"""

import argparse
import os
import tempfile
import traceback

import gradio as gr
import nibabel as nib
import numpy as np

from inference import run_dcrnet


def _make_slice_figure(nii_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vol = nib.load(nii_path).get_fdata(dtype=np.float32)
    vmin, vmax = np.percentile(vol, [2, 98])
    vol_n = np.clip((vol - vmin) / max(vmax - vmin, 1e-6), 0, 1)

    sl = vol_n[:, :, vol_n.shape[2] // 2].T
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    ax.imshow(sl, cmap="gray", origin="lower", aspect="equal")
    ax.set_title("Axial (middle)", fontsize=12)
    ax.axis("off")
    fig.tight_layout(pad=0.5)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    plt.close(fig)
    return buf[:, :, :3].copy()


def reconstruct(
    mat_file,
    checkpoint_file,
    progress=gr.Progress(track_tqdm=True),
):
    if mat_file is None:
        raise gr.Error("Please upload the input .mat file (from Save_Input_Data_For_DCRNet.m).")

    checkpoint_path = checkpoint_file.name if checkpoint_file else None
    output_dir = tempfile.mkdtemp(prefix="dcrnet_out_")

    def _progress(frac, msg):
        progress(frac, desc=msg)

    try:
        real_path, imag_path, mag_path = run_dcrnet(
            mat_path=mat_file.name,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            progress_fn=_progress,
        )
    except Exception:
        raise gr.Error(
            "Reconstruction failed. Check the log for details.\n\n"
            + traceback.format_exc()
        )

    try:
        mag_img = _make_slice_figure(mag_path)
    except Exception:
        mag_img = None

    status = "✅ Reconstruction complete! Download the output NIfTI files below."
    return status, mag_path, real_path, imag_path, mag_img


TITLE = "DCRNet – Accelerated MRI Reconstruction"
DESCRIPTION = """
**Deep Cascade of Recurrent Networks (DCRNet)** for accelerated MRI reconstruction
from undersampled k-space data ([paper](https://doi.org/10.1002/mrm.28680)).

**Input preparation (MATLAB):**
```matlab
Save_Input_Data_For_DCRNet(kspace_data, mask, output_path)
```

**Quick-start:**
1. Prepare the `.mat` input file using the MATLAB script above.
2. Upload the `.mat` file and the DCRNet checkpoint.
3. Click **Run Reconstruction** to get the reconstructed MRI.
"""


def build_ui():
    with gr.Blocks(title=TITLE, theme=gr.themes.Soft()) as demo:
        gr.Markdown(f"# {TITLE}")
        gr.Markdown(DESCRIPTION)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Input")
                mat_file = gr.File(
                    label="Input .mat file (from Save_Input_Data_For_DCRNet.m)",
                    file_types=[".mat"],
                )
                checkpoint_file = gr.File(
                    label="DCRNet checkpoint .pth (optional — uses checkpoints/ default)",
                    file_types=[".pth"],
                )
                run_btn = gr.Button("▶ Run Reconstruction", variant="primary", size="lg")

            with gr.Column(scale=1):
                gr.Markdown("### Results")
                status_box = gr.Textbox(
                    label="Status",
                    lines=2,
                    interactive=False,
                    placeholder="Reconstruction output will appear here …",
                )
                mag_file  = gr.File(label="⬇ Magnitude NIfTI")
                real_file = gr.File(label="⬇ Real part NIfTI")
                imag_file = gr.File(label="⬇ Imaginary part NIfTI")

                gr.Markdown("#### Preview (magnitude, axial middle slice)")
                mag_img = gr.Image(label="Magnitude", show_label=True)

        run_btn.click(
            fn=reconstruct,
            inputs=[mat_file, checkpoint_file],
            outputs=[status_box, mag_file, real_file, imag_file, mag_img],
        )

        gr.Markdown(
            "---\n"
            "**Citation:** Huang W, et al. *Deep low-rank plus sparse network for dynamic MR imaging.* "
            "Magnetic Resonance in Medicine, 2021. "
            "[doi:10.1002/mrm.28680](https://doi.org/10.1002/mrm.28680)\n\n"
            "**Source code:** [github.com/sunhongfu/DCRNet](https://github.com/sunhongfu/DCRNet)"
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DCRNet Gradio server")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--server-name", type=str, default="0.0.0.0")
    args = parser.parse_args()

    demo = build_ui()
    demo.launch(
        share=args.share,
        server_name=args.server_name,
        server_port=args.server_port,
        show_error=True,
        allowed_paths=[tempfile.gettempdir()],
    )
