#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <new>

// Global status reporter
void reportStatus(const std::string &name, bool passed) {
  std::cout << "  " << std::left << std::setw(65) << name;
  if (passed) {
    std::cout << "\033[1;32m[PASSED]\033[0m" << std::endl;
  } else {
    std::cout << "\033[1;31m[FAILED]\033[0m" << std::endl;
  }
}

int main() {
  std::cout
      << "================================================================="
      << std::endl;
  std::cout << "--- WORKBOOK: Basic Memory Offsets (Beginner Level) ---"
            << std::endl;
  std::cout
      << "================================================================="
      << std::endl;

  int passed = 0;
  int total = 0;

  // PROBLEM 1: Add offset to int*
  // Instruction: Given a pointer ptr pointing to the start of arr,
  //              make the pointer 'res' point to index 5 using pointer
  //              addition.
  {
    int arr[10] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
    int *ptr = arr;
    int *res = nullptr;

    // TODO: Make res point to the element at index 5

    bool ok = (res == &arr[5]);
    reportStatus("Problem 1: Add offset to int*", ok);
    if (ok)
      passed++;
    total++;
  }

  // PROBLEM 2: Subtract offset from float*
  // Instruction: Given a pointer ptr pointing to the end of arr (index 9),
  //              make the pointer 'res' point to index 3 using pointer
  //              subtraction.
  {
    float arr[10] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                     6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    float *ptr = &arr[9];
    float *res = nullptr;

    // TODO: Make res point to index 3 using pointer subtraction

    bool ok = (res == &arr[3]);
    reportStatus("Problem 2: Subtract offset from float*", ok);
    if (ok)
      passed++;
    total++;
  }

  // PROBLEM 3: Character Pointer Stepping
  // Instruction: Iterate through the char array 'str' using a pointer.
  //              Find the address of the char 'D' and store it in 'char_ptr'.
  {
    char str[5] = "CUDA";
    char *char_ptr = nullptr;

    // TODO: Write a pointer loop or arithmetic to find and point to the
    // character 'D' in 'str'

    bool ok = (char_ptr != nullptr && *char_ptr == 'D' && char_ptr == &str[2]);
    reportStatus("Problem 3: Character pointer stepping", ok);
    if (ok)
      passed++;
    total++;
  }

  // PROBLEM 4: Base Address Array Decay
  // Instruction: C-style arrays decay to pointers. Point 'ptr' to the start of
  // 'arr'.
  //              Then read the value at index 2 without brackets (using offset
  //              syntax *(ptr + i)) and assign to 'val'.
  {
    int arr[4] = {100, 200, 300, 400};
    int *ptr = nullptr;
    int val = 0;

    // TODO: Point ptr to the base of arr, and retrieve the element at index 2
    // using *(ptr + offset)

    bool ok = (ptr == arr && val == 300);
    reportStatus("Problem 4: Array decay & bracketless dereference", ok);
    if (ok)
      passed++;
    total++;
  }

  // PROBLEM 5: Null Pointer Safety Gate
  // Instruction: You are given a pointer that may or may not be null.
  //              Safely check if the pointer is null. If it is null, set
  //              'is_null' to true. If it is not null, assign the pointed value
  //              to 'val' and set 'is_null' to false.
  {
    int *ptr = nullptr;
    int val = 42;
    bool is_null = false;

    // TODO: Check if ptr is null. Handle null safely to avoid segmentation
    // fault.

    bool ok = (is_null == true && val == 42);

    // Re-run with non-null pointer to verify logic
    int target = 88;
    ptr = &target;

    // TODO: Run the same check logic for the updated ptr

    ok &= (is_null == false && val == 88);
    reportStatus("Problem 5: Null pointer safety gate check", ok);
    if (ok)
      passed++;
    total++;
  }

  // PROBLEM 6: Reference Modification via Pointer
  // Instruction: Bind reference 'ref' to 'val'. Get the address of 'ref' using
  // a pointer 'ptr'.
  //              Then change the value of 'val' to 999 by writing through
  //              'ptr'.
  {
    int val = 100;
    int &ref = val;
    int *ptr = nullptr;

    // TODO: Point ptr to ref, then modify the underlying value to 999 through
    // ptr

    bool ok = (ptr == &val && val == 999);
    reportStatus("Problem 6: Pointer modification of a reference", ok);
    if (ok)
      passed++;
    total++;
  }

  std::cout
      << "\n================================================================="
      << std::endl;
  std::cout << "--- SCORECARD ---" << std::endl;
  std::cout
      << "================================================================="
      << std::endl;
  std::cout << "  Passed: " << passed << " / " << total << " tests."
            << std::endl;
  if (passed == total) {
    std::cout << "\033[1;32m  [STATUS] ALL " << total
              << " TESTS PASSED! \033[0m" << std::endl;
  } else {
    std::cout << "\033[1;31m  [STATUS] INCOMPLETE (" << (total - passed)
              << " tests failed) \033[0m" << std::endl;
  }
  std::cout
      << "================================================================="
      << std::endl;

  return (passed == total) ? 0 : 1;
}
