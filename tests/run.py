import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("ranks", help="Specify ranks to run (e.g. 0,1-3)")
parser.add_argument("--nsys", action="store_true", help="Enable nsys profile")
parser.add_argument("--port", type=int, default=23939, help="Specify distributed port")
parser.add_argument('file', help="Specify test file")
parser.add_argument('args', nargs=argparse.REMAINDER, help="Specify test args")
args = parser.parse_args()

if args.ranks == "all":
    master = os.environ["PADDLE_TRAINERS"].split(",", maxsplit=1)[0]
    rank = int(os.environ["PADDLE_TRAINER_ID"])
    nnodes = int(os.environ["PADDLE_TRAINERS_NUM"])
    pod_index = os.environ["PADDLE_JOB_NAME"] + "-" + os.environ["TRAININGJOB_REPLICA_INDEX"]
else:
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

    master = hosts[ranks[0]]
    rank = ranks.index(pod_index)
    nnodes = len(ranks)

for name in list(os.environ):
    if "PADDLE" in name or "ENDPOINT" in name:
        del os.environ[name]

os.environ["HOME"] = os.getcwd()  # deep_gemm cache
os.environ["SONIC_MOE_CACHE_DIR"] = os.getcwd()
os.environ["FLAGS_use_virtual_memory_auto_growth"] = "1"

log_dir = f"output/trainer.{rank}"

print(end=f"\033[1;32mrank: {rank}, nnodes: {nnodes}, master: {master}, "
      f"trainer: {pod_index}\033[0m\n", flush=True)

cmd = [
    "python", "-um", "paddle.distributed.launch",
    "--log_dir", log_dir,
    "--master", f"{master}:{args.port}",
    "--rank", str(rank), "--nnodes", str(nnodes),
    "--run_mode=collective",
    args.file, *args.args, "grep",
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
