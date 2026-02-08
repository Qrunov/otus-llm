 

#include <cassert>    
#include <cstdint>
#include <iostream>   

 
namespace math {
 
uint64_t phiFunction(uint64_t n) {
    uint64_t result = n;
    for (uint64_t i = 2; i * i <= n; i++) {
        if (n % i != 0)
            continue;
        while (n % i == 0) n /= i;

        result -= result / i;
    }
    if (n > 1)
        result -= result / n;

    return result;
}
}   

 
static void test() {
    assert(math::phiFunction(1) == 1);
    assert(math::phiFunction(2) == 1);
    assert(math::phiFunction(10) == 4);
    assert(math::phiFunction(123456) == 41088);
    assert(math::phiFunction(808017424794) == 263582333856);
    assert(math::phiFunction(3141592) == 1570792);
    assert(math::phiFunction(27182818) == 12545904);

    std::cout << "All tests have successfully passed!\n";
}

 
int main() {
    test();
    return 0;
}
