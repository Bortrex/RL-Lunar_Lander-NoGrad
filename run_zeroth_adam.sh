#!/bin/bash
export JAX_PLATFORMS=cpu

for seed in 123 222 321; do
    echo "Starting seed=$seed"
    python zeroth_order_adam_parallel.py --seed "$seed" 
done

# wait


echo "All runs finished."