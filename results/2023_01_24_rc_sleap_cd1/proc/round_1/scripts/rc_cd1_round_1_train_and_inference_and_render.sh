#!/bin/bash
project_dir=/nancy/projects/dominance_strain_comparison/results/2023_01_24_rc_sleap_cd1/proc/round_1
cd ${project_dir}

sleap-train ${project_dir}/labeled_frames/rc_cd1_round_1_baseline_medium_rf_bottomup.json ${project_dir}/labeled_frames/cd1_combined_10-03-22.mp4.labeled.pkg.slp
model_directory=${project_dir}/models/rc_cd1_round_1_baseline_medium_rf_bottomup

# Day 1
video_directory=/nancy/projects/dominance_strain_comparison/results/2023_01_24_rc_sleap_cd1/data/fixed_reward_competition_D1_videos

# Inference
echo "inference ${video_directory}"
for full_path in ${video_directory}/*.mp4; do
    echo ${full_path}

    dir_name=$(dirname ${full_path})
    file_name=${full_path##*/}
    base_name="${file_name%.mp4}"

    sleap-track ${full_path} --tracking.tracker flow \
    --tracking.similarity centroid --tracking.match greedy \
    --tracking.clean_instance_count 2 \
    --tracking.target_instance_count 2 \
    -m ${model_directory} \
    -o ${project_dir}/predicted_frames/day_1/${base_name}.round_1.predictions.slp
done

# Rendering
echo "rendering ${video_directory}"
for full_path in ${project_dir}/predicted_frames/day_1/*.predictions.slp; do
    echo ${full_path}
    sleap-render ${full_path}
done

# Day 2
video_directory=/nancy/projects/dominance_strain_comparison/results/2023_01_24_rc_sleap_cd1/data/fixed_reward_competition_D2_videos

# Inference
echo "inference ${video_directory}"
for full_path in ${video_directory}/*.mp4; do
    echo ${full_path}

    dir_name=$(dirname ${full_path})
    file_name=${full_path##*/}
    base_name="${file_name%.mp4}"

    sleap-track ${full_path} --tracking.tracker flow \
    --tracking.similarity centroid --tracking.match greedy \
    --tracking.clean_instance_count 2 \
    --tracking.target_instance_count 2 \
    -m ${model_directory} \
    -o ${project_dir}/predicted_frames/day_2/${base_name}.round_1.predictions.slp
done

# Rendering
echo "rendering ${video_directory}"
for full_path in ${project_dir}/predicted_frames/day_2/*.predictions.slp; do
    echo ${full_path}
    sleap-render ${full_path}
done
echo All Done!