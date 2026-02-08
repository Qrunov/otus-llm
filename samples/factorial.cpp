 

#include <cassert>    
#include <cstdint>
#include <iostream>   
 
namespace math {

 
uint64_t factorial(uint8_t n) {
    if (n > 20) {
        throw std::invalid_argument("maximum value is 20\n");
    }
    if (n == 0) {
        return 1;
    }
    return n * factorial(n - 1);
}
}   

 
static void tests() {
    assert(math::factorial(1) == 1);
    assert(math::factorial(0) == 1);
    assert(math::factorial(5) == 120);
    assert(math::factorial(10) == 3628800);
    assert(math::factorial(20) == 2432902008176640000);
    std::cout << "All tests have passed successfully!\n";
}

 
int main() {
    tests();   
    return 0;
}
