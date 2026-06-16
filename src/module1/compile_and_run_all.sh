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
echo "  COMPILING AND RUNNING ALL 60 WORKBOOKS"
echo "================================================================="

# Array of subtopics
subtopics=(
    "1.1_basic_offsets/exercise/beginner_workbook.cpp"
    "1.1_basic_offsets/exercise/intermediate_workbook.cpp"
    "1.1_basic_offsets/exercise/champion_workbook.cpp"
    "1.2_strides_indirection/exercise/beginner_workbook.cpp"
    "1.2_strides_indirection/exercise/intermediate_workbook.cpp"
    "1.2_strides_indirection/exercise/champion_workbook.cpp"
    "1.3_reductions_swaps/exercise/beginner_workbook.cpp"
    "1.3_reductions_swaps/exercise/intermediate_workbook.cpp"
    "1.3_reductions_swaps/exercise/champion_workbook.cpp"
    "1.4_multi_array_strided/exercise/beginner_workbook.cpp"
    "1.4_multi_array_strided/exercise/intermediate_workbook.cpp"
    "1.4_multi_array_strided/exercise/champion_workbook.cpp"
    "1.5_row_col_indexing/exercise/beginner_workbook.cpp"
    "1.5_row_col_indexing/exercise/intermediate_workbook.cpp"
    "1.5_row_col_indexing/exercise/champion_workbook.cpp"
    "1.6_permute_flatten/exercise/beginner_workbook.cpp"
    "1.6_permute_flatten/exercise/intermediate_workbook.cpp"
    "1.6_permute_flatten/exercise/champion_workbook.cpp"
    "1.7_subgrids_padding/exercise/beginner_workbook.cpp"
    "1.7_subgrids_padding/exercise/intermediate_workbook.cpp"
    "1.7_subgrids_padding/exercise/champion_workbook.cpp"
    "1.8_pack_wrap_stencil/exercise/beginner_workbook.cpp"
    "1.8_pack_wrap_stencil/exercise/intermediate_workbook.cpp"
    "1.8_pack_wrap_stencil/exercise/champion_workbook.cpp"
    "1.9_single_array_alloc/exercise/beginner_workbook.cpp"
    "1.9_single_array_alloc/exercise/intermediate_workbook.cpp"
    "1.9_single_array_alloc/exercise/champion_workbook.cpp"
    "1.10_multidim_alignment/exercise/beginner_workbook.cpp"
    "1.10_multidim_alignment/exercise/intermediate_workbook.cpp"
    "1.10_multidim_alignment/exercise/champion_workbook.cpp"
    "1.11_arenas_placements/exercise/beginner_workbook.cpp"
    "1.11_arenas_placements/exercise/intermediate_workbook.cpp"
    "1.11_arenas_placements/exercise/champion_workbook.cpp"
    "1.12_lifetimes_ownership/exercise/beginner_workbook.cpp"
    "1.12_lifetimes_ownership/exercise/intermediate_workbook.cpp"
    "1.12_lifetimes_ownership/exercise/champion_workbook.cpp"
    "1.13_null_bounds_checks/exercise/beginner_workbook.cpp"
    "1.13_null_bounds_checks/exercise/intermediate_workbook.cpp"
    "1.13_null_bounds_checks/exercise/champion_workbook.cpp"
    "1.14_size_padding_punning/exercise/beginner_workbook.cpp"
    "1.14_size_padding_punning/exercise/intermediate_workbook.cpp"
    "1.14_size_padding_punning/exercise/champion_workbook.cpp"
    "1.15_casts_double_pointers/exercise/beginner_workbook.cpp"
    "1.15_casts_double_pointers/exercise/intermediate_workbook.cpp"
    "1.15_casts_double_pointers/exercise/champion_workbook.cpp"
    "1.16_pointer_relations_copying/exercise/beginner_workbook.cpp"
    "1.16_pointer_relations_copying/exercise/intermediate_workbook.cpp"
    "1.16_pointer_relations_copying/exercise/champion_workbook.cpp"
    "1.17_embeddings_projections/exercise/beginner_workbook.cpp"
    "1.17_embeddings_projections/exercise/intermediate_workbook.cpp"
    "1.17_embeddings_projections/exercise/champion_workbook.cpp"
    "1.18_attention_quantization/exercise/beginner_workbook.cpp"
    "1.18_attention_quantization/exercise/intermediate_workbook.cpp"
    "1.18_attention_quantization/exercise/champion_workbook.cpp"
    "1.19_ml_layer_offsets/exercise/beginner_workbook.cpp"
    "1.19_ml_layer_offsets/exercise/intermediate_workbook.cpp"
    "1.19_ml_layer_offsets/exercise/champion_workbook.cpp"
    "1.20_pooling_masks_cycles/exercise/beginner_workbook.cpp"
    "1.20_pooling_masks_cycles/exercise/intermediate_workbook.cpp"
    "1.20_pooling_masks_cycles/exercise/champion_workbook.cpp"
)

# Loop and compile
for item in "${subtopics[@]}"; do
    dir_name=$(dirname "$item")
    file_name=$(basename "$item")
    chapter_name=$(basename $(dirname "$dir_name"))
    level_name="${file_name%_workbook.cpp}"
    binary_name="${chapter_name}_${level_name}"
    
    echo -n "Compiling ${binary_name}... "
    g++ -std=c++20 -O3 "${item}" -o "${OUTPUT_DIR}/${binary_name}" 2>/dev/null
    
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
    echo -e "${GREEN}  [STATUS] CHAMPION STATUS! ALL 60 WORKBOOKS CONCLUDED SUCCESSFULLY!${NC}"
else
    echo -e "${RED}  [STATUS] WORKBOOKS INCOMPLETE (${#FAILED_WORKBOOKS[@]} workbooks failing/unresolved)${NC}"
    echo "  Failing workbooks:"
    for item in "${FAILED_WORKBOOKS[@]}"; do
        echo "    - ${item}"
    done
fi
echo "================================================================="
