 
#include <cassert>    
#include <cstdint>
#include <iostream>   
 
namespace math {
 
uint64_t power(uint64_t a, uint64_t b, uint64_t c) {
    uint64_t ans = 1;   
    a = a % c;          
    if (a == 0) {
        return 0;   
    }
    while (b > 0) {
         
        if (b & 1) {
            ans = ((ans % c) * (a % c)) % c;
        }
         
        b = b >> 1;   
        a = ((a % c) * (a % c)) % c;
    }
    return ans;
}

}   

 
static void test() {
    uint32_t test_case_1 = math::power(2, 5, 13);
    assert(test_case_1 == 6);
    std::cout << "Test 1 Passed!" << std::endl;

    uint32_t test_case_2 = math::power(14, 7, 15);
    assert(test_case_2 == 14);
    std::cout << "Test 2 Passed!" << std::endl;

    uint64_t test_case_3 = math::power(8, 15, 41);
    assert(test_case_3 == 32);
    std::cout << "Test 3 Passed!" << std::endl;

    uint64_t test_case_4 = math::power(27, 2, 5);
    assert(test_case_4 == 4);
    std::cout << "Test 4 Passed!" << std::endl;

    uint16_t test_case_5 = math::power(7, 3, 6);
    assert(test_case_5 == 1);
    std::cout << "Test 5 Passed!" << std::endl;
}

 
int main() {
    test();   
    return 0;
}
