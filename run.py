"""
DCRNet – Command-line interface
Usage:
    python run.py --input data.mat --output ./results/
    python run.py --input data.mat --checkpoint checkpoints/DCRNet_AF8_new.pth --output ./results/
    python run.py --help
"""

import argparse

from inference import run_dcrnet


def main():
    parser = argparse.ArgumentParser(
        description="DCRNet: Accelerated MRI reconstruction from undersampled k-space.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input",      required=True, metavar="FILE",
                        help=".mat file prepared by Save_Input_Data_For_DCRNet.m "
                             "(keys: mask, inputs_img, inputs_k).")
    parser.add_argument("--checkpoint", metavar="FILE", default=None,
                        help="Path to DCRNet checkpoint .pth file. "
                             "Defaults to checkpoints/DCRNet_AF4.pth (auto-downloaded if absent).")
    parser.add_argument("--output",     metavar="DIR",  default="./dcrnet_output",
                        help="Output directory.")

    args = parser.parse_args()

    real_path, imag_path, mag_path = run_dcrnet(
        mat_path=args.input,
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
    )

    print(f"\nOutputs:")
    print(f"  Real:      {real_path}")
    print(f"  Imaginary: {imag_path}")
    print(f"  Magnitude: {mag_path}")


if __name__ == "__main__":
    main()
