# SzoaDel

Robimy SzpaDeL the superior SpaDe

## Running experiments

```bash
uv run white\ obejcts/color_img.py --k 6
./run_experiments.sh
```

Benchmark + concepts:

```bash
uv run python3 download_real_data.py
bash run_real_benchmark.sh
```

Hyperparameter sweep benchmark + concepts:

```bash
uv run python3 download_real_data.py
uv run python3 run_sweep.py
```
