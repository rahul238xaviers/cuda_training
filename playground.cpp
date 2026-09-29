#include <cstdint>
#include <cstring>
#include <iostream>

int main() {
  // Your playground: write and test code here
  uint8_t two_bytes[2] = {0, 1};
  uint16_t *p16 = reinterpret_cast<uint16_t *>(two_bytes);
  std::cout << *p16 << std::endl;
  return 0;
}
