 
#include <cassert>    
#include <cstdint>
#include <iostream>   

 
namespace math {

 
uint64_t binomialCoeffSum(uint64_t n) {
     
    return (1 << n);
}
}   

 
static void test() {
    int test_case_1 = math::binomialCoeffSum(2);
    assert(test_case_1 == 4);
    std::cout << "Test_case_1 Passed!" << std::endl;

    int test_case_2 = math::binomialCoeffSum(3);
    assert(test_case_2 == 8);
    std::cout << "Test_case_2 Passed!" << std::endl;

    int test_case_3 = math::binomialCoeffSum(4);
    assert(test_case_3 == 16);
    std::cout << "Test_case_3 Passed!" << std::endl;

    int test_case_4 = math::binomialCoeffSum(5);
    assert(test_case_4 == 32);
    std::cout << "Test_case_4 Passed!" << std::endl;

    int test_case_5 = math::binomialCoeffSum(7);
    assert(test_case_5 == 128);
    std::cout << "Test_case_5 Passed!" << std::endl;
}

 
int main() {
    test();   
    return 0;
}
