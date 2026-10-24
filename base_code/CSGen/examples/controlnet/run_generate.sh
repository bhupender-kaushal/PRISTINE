#!/usr/bin/env bash
set -euo pipefail

: "${BASE_MODEL:?Set BASE_MODEL to the Stable Diffusion 3.5 Medium directory or model id}"
: "${TRAINED_MODEL_DIR:?Set TRAINED_MODEL_DIR to the directory containing controlnet/ and hpcs/}"
: "${CONTROL_IMAGE:?Set CONTROL_IMAGE to one mask/width/sketch/edge/partial-sketch image}"
: "${DOMAIN:?Set DOMAIN to the curvilinear domain name used in [DOMAIN=...]}"
: "${RULE_PROMPT:?Set RULE_PROMPT to the rule-based geometric descriptor}"
: "${VLM_PROMPT:?Set VLM_PROMPT to the VLM contextual descriptor}"
: "${OUTPUT_IMAGE:?Set OUTPUT_IMAGE to the output PNG path}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/generate.py" \
  --pretrained_model_name_or_path "${BASE_MODEL}" \
  --trained_model_dir "${TRAINED_MODEL_DIR}" \
  --control_image "${CONTROL_IMAGE}" \
  --domain "${DOMAIN}" \
  --rule_prompt "${RULE_PROMPT}" \
  --vlm_prompt "${VLM_PROMPT}" \
  --output "${OUTPUT_IMAGE}" \
  "$@"
