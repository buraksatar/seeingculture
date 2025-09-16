#!/bin/bash
DATA_PATH="your_data_path.csv"
MAX_QUESTIONS_PER_KEYWORD=15
MAX_CHOICE_USAGE_REUSE=25
TOP_K=30
TESTNAME="your_test_name"

INDEX_PATH="your_index_path"

python -m src.mcq_generator_v6 \
    --data_path "$DATA_PATH" \
    --max_questions "$MAX_QUESTIONS_PER_KEYWORD" \
    --max_choice_usage "$MAX_CHOICE_USAGE_REUSE" \
    --top_k "$TOP_K" \
    --testname "$TESTNAME" \
    --index_path "$INDEX_PATH" \