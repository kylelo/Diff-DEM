#!/bin/bash

python="/home/kylelo/anaconda3/envs/Diff-DEM/bin/python"

base_src_dir="./dataset/norway_dem/benchmark"

resume_state="./pretrained/760"

step=1

difficulty=("64-96" "96-128" "128-160")

for diff in "${difficulty[@]}"; do

    out_dir="${diff}"

    ${python} run.py -p test -c config/dem_completion.json -o ${out_dir} -rs ${resume_state} \
                    --data_root ${base_src_dir}/benchmark_gt.flist \
                    --mask_root ${base_src_dir}/mask_${diff}.flist \
                    --scale_factor 1 \
                    --n_timestep ${step}
done

for diff in "${difficulty[@]}"; do
    echo $diff

    out_dir="./experiments/${diff}"

    ${python} ./data/util/tif_metric.py \
                --gt_tif_dir ${base_src_dir}/gt \
                --mask_dir ${base_src_dir}/mask/${diff}/ \
                --algo_dir ${out_dir}/results/test/0/ \
                --normalize # Only required by Diff-DEM! Disable if evaluating other algorthims.
done