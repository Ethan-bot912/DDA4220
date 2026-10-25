import hashlib
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torchvision
from stage_b_damage.train_baseline import build_model

OUT = ROOT / "reports" / "speed_io_cpu_20261025" / "init_proof.txt"
lines = []
def p(*a):
    lines.append(" ".join(str(x) for x in a))

def hash_state(model):
    h = hashlib.sha256()
    for name, t in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def first_forward(model, device="cpu"):
    torch.manual_seed(1234)
    x = torch.randn(8, 3, 128, 128)
    model = model.to(device).eval()
    with torch.no_grad():
        y = model(x.to(device))
    return y.detach().cpu(), x

p("=" * 70)
p("environment")
p("=" * 70)
p("python      ", platform.python_version())
p("torch       ", torch.__version__)
p("torchvision ", torchvision.__version__)
p("cuda avail  ", torch.cuda.is_available())
if torch.cuda.is_available():
    p("cuda device ", torch.cuda.get_device_name(0))
    p("cudnn       ", torch.backends.cudnn.version())
p("")

p("=" * 70)
p("TEST 1: does weight init use the CUDA RNG?")
p("=" * 70)
torch.manual_seed(42)
m1 = build_model("resnet18", mode="post")
h1 = hash_state(m1)

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(9999)
m2 = build_model("resnet18", mode="post")
h2 = hash_state(m2)

p("init hash (cpu seed 42)                  :", h1)
p("init hash (cpu seed 42 + cuda seed 9999) :", h2)
p("identical                                :", h1 == h2)
p("=> init", "does NOT use" if h1 == h2 else "DOES use", "the CUDA RNG")
p("")

p("=" * 70)
p("TEST 2: first-batch forward (CPU)")
p("=" * 70)
torch.manual_seed(42)
model = build_model("resnet18", mode="post")
y_cpu, x = first_forward(model, "cpu")
p("input  sha256 :", hashlib.sha256(x.numpy().tobytes()).hexdigest())
p("output shape  :", tuple(y_cpu.shape))
p("output sha256 :", hashlib.sha256(y_cpu.numpy().tobytes()).hexdigest())
p("output sum    : %.17g" % float(y_cpu.sum()))
p("output[0,:4]  :", ["%.17g" % v for v in y_cpu[0, :4].tolist()])
p("")

if torch.cuda.is_available():
    y_gpu, _ = first_forward(model, "cuda")
    d = (y_cpu - y_gpu).abs()
    p("max |cpu-gpu| : %.6e" % float(d.max()))
    p("mean|cpu-gpu| : %.6e" % float(d.mean()))
    p("bitwise equal :", torch.equal(y_cpu, y_gpu))
else:
    p("No CUDA here -> CPU/GPU forward diff cannot be measured on this box.")
    p("Run this same script on the RTX 4060 machine.")
p("")

p("=" * 70)
p("TEST 3: rebuild reproducibility")
p("=" * 70)
torch.manual_seed(42)
m3 = build_model("resnet18", mode="post")
y3, _ = first_forward(m3, "cpu")
p("same hash on rebuild :",
  hashlib.sha256(y3.numpy().tobytes()).hexdigest()
  == hashlib.sha256(y_cpu.numpy().tobytes()).hexdigest())

OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("WROTE", OUT)
