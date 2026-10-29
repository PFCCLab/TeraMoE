import sys
import argparse


TICKS = [1, 1.005, 1.01, 1.02, 1.05, 1.1, 1.2, 1.5, 2, 3, 4, 6, 8, 1e9]


def main(args):
    base_logs = []
    tera_logs = []
    cur_logs = None

    with open(args.path) as f:
        for line in f:
            if line.startswith("LONG_RUN fleet"):
                cur_logs = base_logs = []
                continue
            if line.startswith("LONG_RUN teramoe"):
                cur_logs = tera_logs = []
                continue
            if cur_logs is None:
                continue
            try:
                i, fwd, bwd, *ratio = line.split(maxsplit=3)
                i = int(i)
                fwd = float(fwd)
                bwd = float(bwd)
                ratio = float(*ratio) if ratio else None
                cur_logs.append((i, fwd, bwd, ratio))
            except:
                cur_logs = None

    print("collected base_logs:", len(base_logs), file=sys.stderr)
    print("collected tera_logs:", len(tera_logs), file=sys.stderr)

    bins = [[[] for _ in TICKS] for _ in range(4)]

    for (_, base_fwd, base_bwd, _), (_, tera_fwd, tera_bwd, ratio) in zip(base_logs, tera_logs):
        for bin_idx in range(len(TICKS) - 1):
            if ratio >= TICKS[bin_idx] and ratio < TICKS[bin_idx + 1]:
                break
        for i, value in enumerate([base_fwd, tera_fwd, base_bwd, tera_bwd]):
            bins[i][bin_idx].append(value)

    print("", "base_fwd", "tera_fwd", "base_bwd", "tera_bwd", "count", sep="\t")

    for bin_idx, tick in enumerate(TICKS):
        print(tick, end="\t")
        for i in range(4):
            bin_ = bins[i][bin_idx]
            if bin_:
                print(sum(bin_) / len(bin_), end="\t")
            else:
                print(end="\t")
        print(len(bin_))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()

    main(args)
