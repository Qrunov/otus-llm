 
#include <cassert>    
#include <cstdint>
#include <iostream>   

 
namespace math {
 
bool magic_number(const uint64_t &n) {
    if (n <= 0) {
        return false;
    }
     
    uint64_t result = n % 9;
     
    if (result == 1) {
        return true;
    } else {
        return false;
    }
}
}   

 
static void tests() {
    std::cout << "Test 1:\t n=60\n";
    assert(math::magic_number(60) == false);
    std::cout << "passed\n";

    std::cout << "Test 2:\t n=730\n";
    assert(math::magic_number(730) == true);
    std::cout << "passed\n";

    std::cout << "Test 3:\t n=0\n";
    assert(math::magic_number(0) == false);
    std::cout << "passed\n";

    std::cout << "Test 4:\t n=479001600\n";
    assert(math::magic_number(479001600) == false);
    std::cout << "passed\n";

    std::cout << "Test 5:\t n=-35\n";
    assert(math::magic_number(-35) == false);
    std::cout << "passed\n";
}

 
int main() {
    tests();   
    return 0;
}
