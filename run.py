"""
DCRNet – Command-line interface
Usage:
    python run.py --config config.yaml
    python run.py --input data.mat --output ./results/
    python run.py --config config.yaml --output ./other/   # CLI overrides config
    python run.py --help
"""

import argparse

import yaml

from inference import run_dcrnet


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", metavar="FILE")
    known, _ = pre.parse_known_args()
    config = _load_config(known.config) if known.config else {}

    parser = argparse.ArgumentParser(
        description="DCRNet: Accelerated MRI reconstruction from undersampled k-space.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config",     metavar="FILE",
                        help="YAML config file. CLI arguments override config values.")
    parser.add_argument("--input",      metavar="FILE",
                        help=".mat file from Save_Input_Data_For_DCRNet.m.")
    parser.add_argument("--checkpoint", metavar="FILE", default=None,
                        help="DCRNet checkpoint .pth (auto-downloaded if omitted).")
    parser.add_argument("--output",     metavar="DIR",  default="./dcrnet_output",
                        help="Output directory.")
    parser.set_defaults(**config)
    args = parser.parse_args()

    if not args.input:
        parser.error("--input is required (or set 'input' in config.yaml).")

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
