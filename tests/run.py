import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("ranks", help="Specify ranks to run (e.g. 0,1-3)")
parser.add_argument("--nsys", action="store_true", help="Enable nsys profile")
parser.add_argument('file', help="Specify test file")
parser.add_argument('args', nargs=argparse.REMAINDER, help="Specify test args")
args = parser.parse_args()

ranks = set()
for r in args.ranks.split(","):
    if r.isdigit():
        ranks.add(int(r))
    else:
        s, e = r.split("-", maxsplit=1)
        ranks.update(range(int(s), int(e) + 1))
ranks = sorted(ranks)

pod_index = int(os.environ["POD_INDEX"])
if pod_index not in ranks:
    exit()

with open("/root/paddlejob/workspace/hostfile") as f:
    hosts = [line.split(maxsplit=1)[0] for line in f]

for name in list(os.environ):
    if "PADDLE" in name or "ENDPOINT" in name:
        del os.environ[name]

os.environ["HOME"] = os.getcwd()  # deep_gemm cache
os.environ["FLAGS_use_virtual_memory_auto_growth"] = "1"

master = hosts[ranks[0]]
port = 23939
rank = ranks.index(pod_index)
nnodes = len(ranks)
log_dir = f"output/trainer.{rank}"

print(end=f"\033[1;32mrank: {rank}, nnodes: {nnodes}, master: {master}, "
      f"trainer: {pod_index}\033[0m\n", flush=True)

cmd = [
    "python", "-um", "paddle.distributed.launch",
    "--log_dir", log_dir,
    "--master", f"{master}:{port}",
    "--rank", str(rank), "--nnodes", str(nnodes),
    "--run_mode=collective",
    args.file, *args.args,
]

if args.nsys:
    cmd = [
        "nsys", "profile",
        "-t", "cuda,nvtx", "-x", "true",
        "--capture-range=cudaProfilerApi",
        "--force-overwrite=true",
        "--cuda-event-trace=false",
        "-o", f"deepep_trainer_{rank}",
        *cmd,
    ]

os.execvp(cmd[0], cmd)
