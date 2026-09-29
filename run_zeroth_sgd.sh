#!/bin/bash
export JAX_PLATFORMS=cpu

for seed in 1234 3210; do
    echo "Starting seed=$seed"
    python zeroth_order_sgd.py --seed "$seed" &
done

wait

echo "All runs finished."