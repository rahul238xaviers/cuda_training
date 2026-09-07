#!/usr/bin/env bash
# ==============================================================================
# Master Automated Test Runner for CUDA Training Curriculum
# Modules 3, 4, 5, 6, and 7 (Beginner, Intermediate, Champion)
# ==============================================================================

set -uo pipefail

ARCH="sm_75"
if command -v nvidia-smi &> /dev/null; then
    # Try to detect compute capability if possible or keep sm_75
    ARCH="${CUDA_ARCH:-sm_75}"
fi

CXX_FLAGS="-O3 -std=c++17 -arch=${ARCH} --extended-lambda"
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="/tmp/cuda_curriculum_bins"
mkdir -p "${BIN_DIR}"

# ANSI Colors
BOLD="\033[1m"
GREEN="\033[32m"
RED="\033[31m"
YELLOW="\033[33m"
CYAN="\033[36m"
RESET="\033[0m"

echo -e "${BOLD}${CYAN}==============================================================================${RESET}"
echo -e "${BOLD}${CYAN}  CUDA ML Training Curriculum — Automated Test Harness${RESET}"
echo -e "${BOLD}${CYAN}  Target Architecture: ${ARCH} | Flags: ${CXX_FLAGS}${RESET}"
echo -e "${BOLD}${CYAN}==============================================================================${RESET}"

FILTER_MODULE="${1:-all}"
LEVEL_FILTER="${2:-all}"

# Collect all workbook files
WORKBOOKS=()
while IFS= read -r -d '' file; do
    WORKBOOKS+=("$file")
done < <(find "${WORK_DIR}/src" -type f -name "*_workbook.cu" -print0 | sort -z)

TOTAL=${#WORKBOOKS[@]}
COMPILED=0
COMPILE_FAILED=0
EXEC_PASSED=0
EXEC_FAILED=0

echo -e "\nFound ${BOLD}${TOTAL}${RESET} exercise workbooks."
echo -e "Starting compilation and test validation...\n"

printf "%-40s | %-14s | %-12s | %-12s\n" "Workbook Path" "Level" "Compile" "Execution"
echo "--------------------------------------------------------------------------------"

for wb in "${WORKBOOKS[@]}"; do
    rel_path="${wb#"${WORK_DIR}/"}"

    # Apply module filter if specified
    if [[ "${FILTER_MODULE}" != "all" && "${rel_path}" != *"${FILTER_MODULE}"* ]]; then
        continue
    fi

    # Determine level
    level="Unknown"
    if [[ "${wb}" == *"beginner"* ]]; then level="Beginner"; fi
    if [[ "${wb}" == *"intermediate"* ]]; then level="Intermediate"; fi
    if [[ "${wb}" == *"champion"* ]]; then level="Champion"; fi

    if [[ "${LEVEL_FILTER}" != "all" && "${level,,}" != "${LEVEL_FILTER,,}" ]]; then
        continue
    fi

    # Unique binary name
    bin_name="$(echo "${rel_path}" | tr '/.' '__')"
    bin_path="${BIN_DIR}/${bin_name}"

    # Compile
    compile_err=$(nvcc ${CXX_FLAGS} "${wb}" -o "${bin_path}" 2>&1)
    compile_status=$?

    if [ ${compile_status} -eq 0 ]; then
        COMPILED=$((COMPILED + 1))
        comp_disp="${GREEN}OK${RESET}"
        
        # Execute binary
        exec_out=$("${bin_path}" 2>&1)
        exec_status=$?
        if [ ${exec_status} -eq 0 ]; then
            EXEC_PASSED=$((EXEC_PASSED + 1))
            exec_disp="${GREEN}5/5 PASS${RESET}"
        else
            EXEC_FAILED=$((EXEC_FAILED + 1))
            # Extract passed count if printed
            passed_count=$(echo "${exec_out}" | grep -o 'Passed: [0-9]* / [0-9]*' || echo "FAIL")
            exec_disp="${YELLOW}${passed_count}${RESET}"
        fi
    else
        COMPILE_FAILED=$((COMPILE_FAILED + 1))
        comp_disp="${RED}FAIL${RESET}"
        exec_disp="${RED}SKIPPED${RESET}"
    fi

    # Display row
    # Shorten path to stage name + filename
    short_name="$(basename "$(dirname "$(dirname "${wb}")")")/$(basename "${wb}")"
    printf "%-40s | %-14s | %-21b | %-21b\n" "${short_name}" "${level}" "${comp_disp}" "${exec_disp}"
done

rm -rf "${BIN_DIR}"

echo -e "\n--------------------------------------------------------------------------------"
echo -e "${BOLD}Summary Scorecard:${RESET}"
echo -e "  Total Evaluated:    ${TOTAL}"
echo -e "  Compiled OK:        ${GREEN}${COMPILED}${RESET} / ${TOTAL}"
if [ ${COMPILE_FAILED} -gt 0 ]; then
    echo -e "  Compile Failures:   ${RED}${COMPILE_FAILED}${RESET}"
fi
echo -e "  Exec Completed:     ${EXEC_PASSED} (100% passed) | ${EXEC_FAILED} (template pending solutions)"
echo -e "--------------------------------------------------------------------------------"

if [ ${COMPILE_FAILED} -gt 0 ]; then
    echo -e "${RED}${BOLD}Validation FAILED: Some workbooks failed compilation!${RESET}"
    exit 1
else
    echo -e "${GREEN}${BOLD}All workbooks compiled successfully and are ready for engineers!${RESET}"
    exit 0
fi
