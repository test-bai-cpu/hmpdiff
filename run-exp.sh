#!/usr/bin/env bash
set -euo pipefail

# PARAM_SETUP_V="1"

# export PYTHONUNBUFFERED=1

# for PRED_LEN in 20 40 60; do
#     for SIGMA in 0.1 0.2 0.3 0.4 0.5 1.0; do
#         IF_UT=1
#         for IF_MOD in 0 1; do
#             for IF_DIV in 0 1; do
#                 for IF_SMOOTH in 0 1; do

#                     VERSION="V${PARAM_SETUP_V}-pred${PRED_LEN}-sigma${SIGMA}-ut${IF_UT}-mod${IF_MOD}-div${IF_DIV}-smooth${IF_SMOOTH}"

#                     echo "=== Running ==="
#                     echo "  VERSION: ${VERSION}"
#                     echo "============="

#                     python3 -u train_k.py \
#                     "${PARAM_SETUP_V}" "${PRED_LEN}" "${SIGMA}" \
#                     "${IF_UT}" "${IF_MOD}" "${IF_DIV}" "${IF_SMOOTH}" \
#                     2>&1 | tee "logs/${VERSION}.log"
#                 done
#             done
#         done
#     done
# done


# PARAM_SETUP_V="2-fix-t"
# PRED_LEN=20
# SIGMA=0.2
# IF_UT=1
# IF_MOD=1
# IF_DIV=1
# IF_SMOOTH=1

# VERSION="V${PARAM_SETUP_V}-pred${PRED_LEN}-sigma${SIGMA}-ut${IF_UT}-mod${IF_MOD}-div${IF_DIV}-smooth${IF_SMOOTH}"

# echo "=== Running ==="
# echo "  VERSION: ${VERSION}"
# echo "============="

# python3 -u train_k.py \
# "${PARAM_SETUP_V}" "${PRED_LEN}" "${SIGMA}" \
# "${IF_UT}" "${IF_MOD}" "${IF_DIV}" "${IF_SMOOTH}" \
# 2>&1 | tee "logs/${VERSION}.log"


PARAM_SETUP_V="2-fix-t"
PRED_LEN=20
SIGMA=0.2
IF_UT=0
IF_MOD=1
IF_DIV=1
IF_SMOOTH=1

VERSION="V${PARAM_SETUP_V}-pred${PRED_LEN}-sigma${SIGMA}-ut${IF_UT}-mod${IF_MOD}-div${IF_DIV}-smooth${IF_SMOOTH}"

echo "=== Running ==="
echo "  VERSION: ${VERSION}"
echo "============="

python3 -u train_k.py \
"${PARAM_SETUP_V}" "${PRED_LEN}" "${SIGMA}" \
"${IF_UT}" "${IF_MOD}" "${IF_DIV}" "${IF_SMOOTH}" \
2>&1 | tee "logs/${VERSION}.log"


PARAM_SETUP_V="2-fix-t"
PRED_LEN=20
SIGMA=0.2
IF_UT=1
IF_MOD=0
IF_DIV=1
IF_SMOOTH=1

VERSION="V${PARAM_SETUP_V}-pred${PRED_LEN}-sigma${SIGMA}-ut${IF_UT}-mod${IF_MOD}-div${IF_DIV}-smooth${IF_SMOOTH}"

echo "=== Running ==="
echo "  VERSION: ${VERSION}"
echo "============="

python3 -u train_k.py \
"${PARAM_SETUP_V}" "${PRED_LEN}" "${SIGMA}" \
"${IF_UT}" "${IF_MOD}" "${IF_DIV}" "${IF_SMOOTH}" \
2>&1 | tee "logs/${VERSION}.log"


PARAM_SETUP_V="2-fix-t"
PRED_LEN=20
SIGMA=0.2
IF_UT=0
IF_MOD=0
IF_DIV=1
IF_SMOOTH=1

VERSION="V${PARAM_SETUP_V}-pred${PRED_LEN}-sigma${SIGMA}-ut${IF_UT}-mod${IF_MOD}-div${IF_DIV}-smooth${IF_SMOOTH}"

echo "=== Running ==="
echo "  VERSION: ${VERSION}"
echo "============="

python3 -u train_k.py \
"${PARAM_SETUP_V}" "${PRED_LEN}" "${SIGMA}" \
"${IF_UT}" "${IF_MOD}" "${IF_DIV}" "${IF_SMOOTH}" \
2>&1 | tee "logs/${VERSION}.log"