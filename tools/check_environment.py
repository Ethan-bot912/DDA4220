"""Record and verify this project's installed GPU environment without downloads."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import sys
from pathlib import Path

import torch

EXPECTED = {'torch': '2.7.1+cu128', 'torchvision': '0.22.1+cu128', 'numpy': '2.0.2',
            'rasterio': '1.4.3', 'matplotlib': '3.9.2'}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('output/environment.json'))
    args = parser.parse_args()
    installed = {name: importlib.metadata.version(name) for name in EXPECTED}
    mismatches = {name: {'expected': value, 'installed': installed[name]}
                  for name, value in EXPECTED.items() if installed[name] != value}
    ready = torch.cuda.is_available()
    result = {'python': platform.python_version(), 'executable': sys.executable,
              'versions': installed, 'version_mismatches': mismatches,
              'cuda_available': ready, 'torch_cuda_runtime': torch.version.cuda,
              'project_drive_free_bytes': shutil.disk_usage(Path.cwd()).free}
    if ready:
        x = torch.tensor([[1., 2.], [3., 4.]], device='cuda')
        assert torch.equal((x @ x).cpu(), torch.tensor([[7., 10.], [15., 22.]]))
        result.update(gpu=torch.cuda.get_device_name(0),
                      gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
                      gpu_matmul_verified=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    return 0 if ready and not mismatches else 1


if __name__ == '__main__':
    raise SystemExit(main())
