#!/bin/bash
# Wait for AST to finish, then run BEATs
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export LD_LIBRARY_PATH=/home/haipd/miniconda3/lib/python3.13/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1

cd /home/haipd/Parallel_Computing_on_FPGA/python/rsc-ensemble-kd

# Wait for AST to finish
echo "Waiting for AST training to finish..."
while ! grep -q "finished" logs/train_AST_AudioSet_1.log 2>/dev/null; do
    sleep 10
done
echo "AST finished! Starting BEATs..."

# Run BEATs
bash scripts/train_BEATs_teacher.sh
