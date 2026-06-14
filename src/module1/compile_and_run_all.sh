#!/bin/bash

# ANSI colors
GREEN='\033[1;32m'
RED='\033[1;31m'
NC='\033[0m'

# Find the root of the workspace
WORKSPACE_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
OUTPUT_DIR="${WORKSPACE_ROOT}/output"
mkdir -p "${OUTPUT_DIR}"

# Run relative to the script's directory
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "${SCRIPT_DIR}"

TOTAL_PASSED=0
TOTAL_TESTS=0
FAILED_WORKBOOKS=()

echo "================================================================="
echo "  COMPILING AND RUNNING ALL 20 WORKBOOKS (100 PROBLEMS)"
echo "================================================================="

# Array of subtopics
subtopics=(
    "1.1_basic_offsets/basic_offsets_workbook.cpp"
    "1.2_strides_indirection/strides_indirection_workbook.cpp"
    "1.3_reductions_swaps/reductions_swaps_workbook.cpp"
    "1.4_multi_array_strided/multi_array_strided_workbook.cpp"
    "1.5_row_col_indexing/row_col_indexing_workbook.cpp"
    "1.6_permute_flatten/permute_flatten_workbook.cpp"
    "1.7_subgrids_padding/subgrids_padding_workbook.cpp"
    "1.8_pack_wrap_stencil/pack_wrap_stencil_workbook.cpp"
    "1.9_single_array_alloc/single_array_alloc_workbook.cpp"
    "1.10_multidim_alignment/multidim_alignment_workbook.cpp"
    "1.11_arenas_placements/arenas_placements_workbook.cpp"
    "1.12_lifetimes_ownership/lifetimes_ownership_workbook.cpp"
    "1.13_null_bounds_checks/null_bounds_checks_workbook.cpp"
    "1.14_size_padding_punning/size_padding_punning_workbook.cpp"
    "1.15_casts_double_pointers/casts_double_pointers_workbook.cpp"
    "1.16_pointer_relations_copying/pointer_relations_copying_workbook.cpp"
    "1.17_embeddings_projections/embeddings_projections_workbook.cpp"
    "1.18_attention_quantization/attention_quantization_workbook.cpp"
    "1.19_ml_layer_offsets/ml_layer_offsets_workbook.cpp"
    "1.20_pooling_masks_cycles/pooling_masks_cycles_workbook.cpp"
)

# Loop and compile
for item in "${subtopics[@]}"; do
    dir_name=$(dirname "$item")
    file_name=$(basename "$item")
    binary_name="${file_name%.cpp}"
    
    echo -n "Compiling ${binary_name} inside ${dir_name}... "
    g++ -std=c++20 -O3 "${dir_name}/${file_name}" -o "${OUTPUT_DIR}/${binary_name}" 2>/dev/null
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}[COMPILE FAILED]${NC}"
        FAILED_WORKBOOKS+=("${binary_name} (compile)")
        continue
    else
        echo -e "${GREEN}[OK]${NC}"
    fi
    
    # Run the workbook and parse scorecard
    output=$("${OUTPUT_DIR}/${binary_name}")
    ret_code=$?
    
    # Parse output: "Passed: X / Y tests."
    passed_line=$(echo "$output" | grep "Passed:")
    passed_count=$(echo "$passed_line" | awk '{print $2}')
    total_count=$(echo "$passed_line" | awk '{print $4}')
    
    if [ -z "$passed_count" ]; then
        passed_count=0
        total_count=5
    fi
    
    TOTAL_PASSED=$((TOTAL_PASSED + passed_count))
    TOTAL_TESTS=$((TOTAL_TESTS + total_count))
    
    if [ $ret_code -ne 0 ]; then
        FAILED_WORKBOOKS+=("${binary_name}")
    fi
done

echo ""
echo "================================================================="
echo "  MASTER SCORECARD: MODULE 1 C++ FUNDAMENTALS"
echo "================================================================="
echo -e "  Overall Progress: ${GREEN}${TOTAL_PASSED}${NC} / ${GREEN}${TOTAL_TESTS}${NC} tests passed."
echo "================================================================="

if [ ${#FAILED_WORKBOOKS[@]} -eq 0 ]; then
    echo -e "${GREEN}  [STATUS] CHAMPION STATUS! ALL 20 WORKBOOKS CONCLUDED SUCCESSFULLY!${NC}"
else
    echo -e "${RED}  [STATUS] WORKBOOKS INCOMPLETE (${#FAILED_WORKBOOKS[@]} workbooks failing/unresolved)${NC}"
    echo "  Failing workbooks:"
    for item in "${FAILED_WORKBOOKS[@]}"; do
        echo "    - ${item}"
    done
fi
echo "================================================================="
