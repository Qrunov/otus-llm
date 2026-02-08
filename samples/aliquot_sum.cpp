 

#include <cassert>    
#include <cstdint>
#include <iostream>   

 
namespace math {

 
uint64_t aliquot_sum(const uint64_t num) {
    if (num == 0 || num == 1) {
        return 0;   
    }

    uint64_t sum = 0;

    for (uint64_t i = 1; i <= num / 2; i++) {
        if (num % i == 0) {
            sum += i;
        }
    }

    return sum;
}
}   

 
static void test() {
     
    assert(math::aliquot_sum(10) == 8);
     
    assert(math::aliquot_sum(15) == 9);
     
    assert(math::aliquot_sum(1) == 0);
     
    assert(math::aliquot_sum(97) == 1);

    std::cout << "All the tests have successfully passed!\n";
}

 
int main() {
    test();   
    return 0;
}
