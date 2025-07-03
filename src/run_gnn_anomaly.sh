#!/bin/bash

echo "Started GNN Anomaly Detection..."

SCRIPT_PATH="../Agent/GNN_Anomaly_Detection-master/src/main.py"

# ========== 2017 Data ==========
INPUT_DIR_2017="../Agent/data/2017"
OUTPUT_DIR_2017="../data/2017"
mkdir -p "$OUTPUT_DIR_2017"

declare -a FILES_2017=(
    "SQL_injection.csv"
    "bruteForce.csv"
    "dos-slowloris.csv"
    "slowhttptest.csv"
    "xss.csv"
)

for FILE in "${FILES_2017[@]}"; do
    for SIZE in 4000 2000 1000; do
        INPUT_FILE="${INPUT_DIR_2017}/${FILE}"
        OUTPUT_FILE="${OUTPUT_DIR_2017}/${FILE%.csv}_${SIZE}.txt"

        if [ ! -f "$INPUT_FILE" ]; then
            echo "❌ Input file not found: $INPUT_FILE"
            continue
        fi

        echo "Running 2017 ${FILE%.csv} with size ${SIZE}..."
        python "$SCRIPT_PATH" "$INPUT_FILE" "$SIZE" | grep -vE '^(Accuracy:| *precision| *False| *True| *accuracy| *macro avg| *weighted avg|True Positive Rate|False Positive Rate|Area Under the ROC Curve|ROC Curve Points:|Threshold|Processing Time:|CPU Usage:|Memory Usage:)' > "$OUTPUT_FILE"
    done
done

# ========== CIC-2018 Data ==========
INPUT_DIR_2018="../Agent/data/cic2018TpSend"
OUTPUT_DIR_2018="../data/cic2018TpSend"
mkdir -p "$OUTPUT_DIR_2018"

FILTER_REGEX='^(Accuracy:| *precision| *False| *True| *accuracy| *macro avg| *weighted avg|True Positive Rate|False Positive Rate|Area Under the ROC Curve|ROC Curve Points:|Threshold|Processing Time:|CPU Usage:|Memory Usage:)'

declare -a ATTACKS=(
    "BruteForce-Web:Thurs-22-02-BruteForce-Web-benign.csv:172.31.69.28:18.218.115.60"
    "BruteForce-XSS:Thurs-22-02-BruteForce-XSS-benign.csv:172.31.69.28:18.218.115.60"
    "SQL-Injection:Thurs-22-02-SQL-Injection-benign.csv:172.31.69.28:18.218.115.60"
    "DoS-Slowloris:Thurs-15-02-DoS-Slowloris-benign.csv:172.31.69.25:18.217.165.70"
)

for ATTACK_INFO in "${ATTACKS[@]}"; do
    IFS=':' read -r ATTACK FILE SRC_IP DST_IP <<< "$ATTACK_INFO"

    echo "Started running $ATTACK"

    INPUT_FILE="${INPUT_DIR_2018}/${FILE}"

    if [ ! -f "$INPUT_FILE" ]; then
        echo "❌ Input file not found: $INPUT_FILE"
        continue
    fi

    for SIZE in 4000 2000 1000; do
        OUTPUT_FILE="${OUTPUT_DIR_2018}/${FILE%.csv}_${SIZE}_1.txt"
        mkdir -p "$(dirname "$OUTPUT_FILE")"

        echo "Running ${ATTACK} with size ${SIZE}..."
        python "$SCRIPT_PATH" "$INPUT_FILE" "$SIZE" "$SRC_IP" "$DST_IP" | grep -vE "$FILTER_REGEX" > "$OUTPUT_FILE"
    done
done

echo "✅ Completed all runs."
