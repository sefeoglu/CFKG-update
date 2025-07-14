#!/bin/bash

BASE_PATH="/Users/sefika/phd_projects/tgdk-paper/data/finance/data/cl_task/test"
OUTPUT_BASE="/Users/sefika/phd_projects/tgdk-paper/data/finance/data/shots/random_formatted_test_flan-t5"
# INITIAL="/Users/sefika/phd_projects/tgdk-paper/data/finance/data/cl_task"
# for run in {1..5}; do
#   cp -r "$INITIAL/run_${run}/task_1/train.json" "$BASE_PATH/run_${run}/task_1/train.json"
# done

for run in {1..5}; do
  for task in {1..5}; do
    for split in test; do
      INPUT_FILE="$BASE_PATH/run_${run}/task_${task}/${split}.json"
      OUTPUT_FILE="$OUTPUT_BASE/run_${run}/task_${task}/${split}.json"
      
      echo "Running: $INPUT_FILE → $OUTPUT_FILE"
      python prompt_generation.py --path_to_data "$INPUT_FILE" --output_folder "$OUTPUT_FILE" --format_type flan-t5
    done
  done
done
