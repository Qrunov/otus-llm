 
#include <cassert>    
#include <cstdint>
#include <iostream>   

 
namespace math {
 
bool is_factorial(uint64_t n) {
    if (n <= 0) {   
        return false;
    }

     
    int i = 2;
    while (n % i == 0) {
        n = n / i;
        i++;
    }

     
    return (n == 1);
}
}   

 
static void tests() {
    assert(math::is_factorial(50) == false);
    assert(math::is_factorial(720) == true);
    assert(math::is_factorial(0) == false);
    assert(math::is_factorial(1) == true);
    assert(math::is_factorial(479001600) == true);
    assert(math::is_factorial(-24) == false);

    std::cout << "All tests have successfully passed!" << std::endl;
}

 
int main() {
    tests();   
    return 0;
}
