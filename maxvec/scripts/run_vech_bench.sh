#!/bin/bash
# Run the Vec-H queries with the authors' results/run_vsds.sh, unmodified, on one GPU node.
# - Data: synthetic SF1 with real embedding dims (reviews 1024-d, images 1152-d), generated
#   on node-local disk if not there yet (maxbench requires exactly these dims). TPC-H tables
#   are written as CSV, as in the real Vec-H dataset: MaxVec reads them with its float64
#   schema. The synthetic generator's own Parquet TPC-H uses decimal128, which breaks GPU
#   filters ("AST expression ... non-matching operand types") and changes CPU arithmetic.
# - run_vsds.sh expects to run from maxvec/ with ./build/Release and
#   ./tests/vsds/data-industrial_and_scientific-sf_1/; a run dir with symlinks provides that.
# - Logs (caliper reports, plans, timings) are copied to ~/vecopt/runs/<jobid>/.
#
#   cd ~/maxvec-build && sbatch ~/vecopt/euler/run_vech_bench.sh
#   REPS=20 CONFIGS="c3:enn:flat c3:ann:cagra" sbatch ~/vecopt/euler/run_vech_bench.sh
#   NO_INDEX_CACHE=1 ...   # needed for GPU runs of review queries on 40 GB GPUs (A100-40GB)
#
# CONFIGS entries are <case>:<enn|ann>:<index filter>, see run_vsds.sh. Cases: c0 = all CPU,
# c1 = GPU exec / CPU storage, c3 = all GPU, c4 = GPU exec with CPU vector search.
#
#SBATCH --job-name=vech-bench
#SBATCH --partition=research
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.log

set -uo pipefail
CONFIGS=${CONFIGS:-"c3:enn:flat c3:ann:cagra c3:ann:ivf1024 c0:enn:flat c0:ann:ivf1024"}
export REPS=${REPS:-10}
K=${K:-100}
JOB=${SLURM_JOB_ID:-manual}

source ~/maxvec-env.sh
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -cs 'A-Za-z0-9' '_' | sed 's/_$//')
SYSTEM=${SYSTEM:-euler-$GPU_NAME}

# 1. Dataset on node-local disk
GEN=~/vecopt/vec-h/dataset-generation
DATA=/tmp/$USER/data/synthetic-sf1-d1024-csv
if [ ! -f "$DATA/.complete" ]; then
    echo "== generate synthetic SF1 (1024/1152-d, TPC-H as CSV) [$(date +%T)]"
    rm -rf "$DATA"
    python -I $GEN/generate_synthetic_vech.py \
        --sf 1 --rev_dim 1024 --img_dim 1152 --output_dir "$DATA" --skip_tpch || exit 1
    python -I $GEN/generate_tpch_tables.py --sf 1 --n_files 1 --output_dir_csv "$DATA" || exit 1
    touch "$DATA/.complete"
fi
ls "$DATA"
du -sh "$DATA"

# 2. Run dir with the layout run_vsds.sh expects
RUN=/tmp/$USER/vech-run-$JOB
mkdir -p "$RUN/build" "$RUN/tests/vsds"
ln -sfn "$MAXVEC_BUILD" "$RUN/build/Release"
ln -sfn "$DATA" "$RUN/tests/vsds/data-industrial_and_scientific-sf_1"
RES=$RUN/results
cd "$RUN"

# 3. Benchmarks
for cfg in $CONFIGS; do
    IFS=: read -r case vs filter <<<"$cfg"
    echo "== $case $vs $filter [$(date +%T)]"
    CUDA_VISIBLE_DEVICES=0 MAXIMUS_OPERATORS_FUSION=0 MAXIMUS_DATASET_API=0 \
        bash ~/vecopt/maxvec/results/run_vsds.sh "$case" 1 "$vs" "$SYSTEM" "$K" "$RES" false "$filter" \
        2>&1 | grep -E "^(Running|Executing|\*\*\*)" | cut -c1-200
done

# 4. Save logs to home
OUT=$HOME/vecopt/runs/$JOB
mkdir -p "$OUT"
cp -r "$RES"/. "$OUT"/
echo "== saved to $OUT [$(date +%T)]"
grep -h -E "DONE|TIMEOUT|failed" "$OUT"/vsds/*/sf_1/progress.log | cut -c1-160
