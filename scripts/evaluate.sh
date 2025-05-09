#!/bin/bash
set -eu

SVN_ROOT="${HOME}/svn"
SCRIPT_ROOT="${SVN_ROOT}/FMEProjects/BreastMARC/Projects/nnDetectionEvaluation/snakemake"
WORKSPACE_DATA_OVERVIEW_ROOT="/Projects/WorkspaceDataOverview"

deps="\
    --with pandas \
    --with numpy \
    --with matplotlib \
    --with seaborn \
    --with scikit-learn \
    --with scipy \
    --exclude-newer=2025-05-09 \
    --python 3.12 \
    --isolated \
    --no-config \
    --no-project \
    --script \
"

RESULT_CSV=${1}
OUTPUT_FOLDER=${2}
REFERENCE_DATE=${3}

RESULT_NAME=$(basename -s .csv ${RESULT_CSV})
OUTPUT_DIR="${OUTPUT_FOLDER}/${RESULT_NAME}"

REFERENCES_JSON="${OUTPUT_FOLDER}/references.json"
DETECTIONS_JSON="${OUTPUT_DIR}/detections.json"
METRICS_JSON="${OUTPUT_DIR}/metrics.json"


[ -d $OUTPUT_DIR ] || mkdir ${OUTPUT_DIR}

# create reference json
uv run $deps "${SCRIPT_ROOT}/create_reference_json.py" \
    "${WORKSPACE_DATA_OVERVIEW_ROOT}/workspace_overview_imagelevel_all_${REFERENCE_DATE}.csv" \
    "${WORKSPACE_DATA_OVERVIEW_ROOT}/workspace_overview_structurelevel_shared_reader_${REFERENCE_DATE}.csv" \
    "${REFERENCES_JSON}"

# load and parse detection csv
uv run $deps "${SCRIPT_ROOT}/load_and_parse_detection_csv.py" "${RESULT_CSV}" "${DETECTIONS_JSON}"

# compute metrics
uv run $deps "${SCRIPT_ROOT}/compute_metrics.py" "${DETECTIONS_JSON}" "${REFERENCES_JSON}" "${METRICS_JSON}"

# create plots
uv run $deps "${SCRIPT_ROOT}/create_plots.py" "${OUTPUT_FOLDER}" "${OUTPUT_FOLDER}/plots"

# create tables
uv run $deps "${SCRIPT_ROOT}/create_tables.py" "${OUTPUT_FOLDER}" "${OUTPUT_FOLDER}/tables"