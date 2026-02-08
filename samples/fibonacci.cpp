 

#include <cstdint>
#include <cassert>    
#include <iostream>   

 
namespace math {
 
namespace fibonacci {
 
uint64_t fibonacci(uint64_t n) {
     
     
    if (n <= 1) {
        return n;
    }

     
    return fibonacci(n - 1) + fibonacci(n - 2);
}
}   
}   

 
static void test() {
    assert(math::fibonacci::fibonacci(0) == 0);
    assert(math::fibonacci::fibonacci(1) == 1);
    assert(math::fibonacci::fibonacci(2) == 1);
    assert(math::fibonacci::fibonacci(3) == 2);
    assert(math::fibonacci::fibonacci(4) == 3);
    assert(math::fibonacci::fibonacci(15) == 610);
    assert(math::fibonacci::fibonacci(20) == 6765);
    std::cout << "All tests have passed successfully!\n";
}

 
int main() {
    test();   
    return 0;
}
