 

#include <cassert>    
#include <cstdint>
#include <iostream>   
 
namespace math {

 
uint64_t largestPower(uint32_t n, const uint16_t& p) {
     
    int x = 0;

     
    while (n) {
        n /= p;
        x += n;
    }
    return x;
}

}   

 
static void test() {
    uint8_t test_case_1 = math::largestPower(5, 2);
    assert(test_case_1 == 3);
    std::cout << "Test 1 Passed!" << std::endl;

    uint16_t test_case_2 = math::largestPower(10, 3);
    assert(test_case_2 == 4);
    std::cout << "Test 2 Passed!" << std::endl;

    uint32_t test_case_3 = math::largestPower(25, 5);
    assert(test_case_3 == 6);
    std::cout << "Test 3 Passed!" << std::endl;

    uint32_t test_case_4 = math::largestPower(27, 2);
    assert(test_case_4 == 23);
    std::cout << "Test 4 Passed!" << std::endl;

    uint16_t test_case_5 = math::largestPower(7, 3);
    assert(test_case_5 == 2);
    std::cout << "Test 5 Passed!" << std::endl;
}

 
int main() {
    test();   
    return 0;
}
