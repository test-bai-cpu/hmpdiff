#!/usr/bin/env bash
set -euo pipefail

PARAM_SETUP_V="14"

export PYTHONUNBUFFERED=1

lambda_mod=1e-3
for PRED_LEN in 20 40 60; do
    for SIGMA in 0.1; do
        for IF_MOD in 0; do
            for K in 5; do
                VERSION="V${PARAM_SETUP_V}-pred${PRED_LEN}-sigma${SIGMA}-mod${IF_MOD}-k${K}-lambdaMod${lambda_mod}"

                echo "=== Running ==="
                echo "  VERSION: ${VERSION}"
                echo "============="

                python3 -u train_k.py \
                "${PARAM_SETUP_V}" "${PRED_LEN}" "${SIGMA}" \
                "${IF_MOD}" "${K}" "${lambda_mod}" \
                2>&1 | tee "logs/${VERSION}.log"
            done
        done
    done
done

