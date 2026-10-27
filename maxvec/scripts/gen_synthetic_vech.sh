#!/bin/bash
# Generate a synthetic Vec-H dataset (real TPC-H tables + reviews/images with random vectors)
# with vec-h/dataset-generation/generate_synthetic_vech.py, into ~/vecopt/data/synthetic-sf<SF>.
# Extra arguments are passed to the generator (e.g. --rev_dim 1024 --img_dim 1152).
#
#   cd ~/maxvec-build && sbatch ~/vecopt/euler/gen_synthetic_vech.sh            # SF1, 128-d
#   SF=10 sbatch ~/vecopt/euler/gen_synthetic_vech.sh --rev_dim 1024 --img_dim 1152
#
#SBATCH --job-name=vech-synth
#SBATCH --partition=research
#SBATCH --constraint=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=%x-%j.log

set -euo pipefail
SF=${SF:-1}
NAME=${NAME:-synthetic-sf$SF}
DEST=$HOME/vecopt/data/$NAME
GEN=$HOME/vecopt/vec-h/dataset-generation/generate_synthetic_vech.py
T=/tmp/$USER/$NAME

source ~/maxvec-env.sh        # python with numpy, pyarrow, duckdb
rm -rf "$T"
echo "== generate SF$SF on $(hostname) [$(date +%T)]"
python -I "$GEN" --sf "$SF" --output_dir "$T" "$@"

echo "== copy to $DEST [$(date +%T)]"
mkdir -p "$DEST"
cp -r "$T"/. "$DEST"/
du -sh "$DEST"
rm -rf "$T"
echo "== done [$(date +%T)]"
