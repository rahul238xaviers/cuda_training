#include <cstdint>
#include <cstring>
#include <iostream>

int main() {
  // Your playground: write and test code here
  uint8_t buffer[12] = {1,2,3,4,5,6,7,8,9,10,11,12};
  uint32_t val_a = 7;

  std::memcpy(buffer, &val_a, sizeof(uint32_t));

  for (int i = 0; i < 12; i++) {
    std::cout << buffer[i] << " ";
  }
  std::cout << std::endl;

  return 0;
}
