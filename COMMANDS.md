# Command Quick Reference

## Interactive Session

Run a faster interactive shell session on `workq`:

```bash
srun --partition=workq --gpus=4 --cpus-per-task=16 --mem=64G --time=02:00:00 --pty bash -l
```

## SLURM Launchers

Zero-shot inference across configured active models.

```bash
bash slurm/zero-shot-inference.sh
```

Finetune-model inference mode (routes to unified inference script).

```bash
bash slurm/model-finetune.sh
```

Evaluate zero-shot prediction files.

```bash
bash slurm/evaluation-stats.sh
```

Evaluate finetune prediction files.

```bash
bash slurm/evaluation-stats-finetune.sh
```

Run self-reflection pass on verified outputs.

```bash
bash slurm/self-reflection.sh
```

Run tokenization ablation for one task family/model setup.

```bash
bash slurm/tokenization-ablation.sh
```

Submit both input/output tokenization-ablation families together.

```bash
bash slurm/run-tokenization-ablation-both.sh
```

Generate execution traces.

```bash
bash slurm/generate-execution-trace.sh
```
