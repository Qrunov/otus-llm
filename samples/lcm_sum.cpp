 

#include <cassert>    
#include <cstdint>
#include <iostream>   
#include <vector>     

 
namespace math {
 
uint64_t lcmSum(const uint16_t& num) {
    uint64_t i = 0, j = 0;
    std::vector<uint64_t> eulerTotient(num + 1);
    std::vector<uint64_t> sumOfEulerTotient(num + 1);

     
    for (i = 1; i <= num; i++) {
        eulerTotient[i] = i;
    }

     
    for (i = 2; i <= num; i++) {
        if (eulerTotient[i] == i) {
            for (j = i; j <= num; j += i) {
                eulerTotient[j] = eulerTotient[j] / i;
                eulerTotient[j] = eulerTotient[j] * (i - 1);
            }
        }
    }

     
    for (i = 1; i <= num; i++) {
        for (j = i; j <= num; j += i) {
            sumOfEulerTotient[j] += eulerTotient[i] * i;
        }
    }

    return ((sumOfEulerTotient[num] + 1) * num) / 2;
}
}   

 
static void test() {
    uint64_t n = 2;
    uint64_t test_1 = math::lcmSum(n);
    assert(test_1 == 4);
    std::cout << "Passed Test 1!" << std::endl;

    n = 5;
    uint64_t test_2 = math::lcmSum(n);
    assert(test_2 == 55);
    std::cout << "Passed Test 2!" << std::endl;

    n = 10;
    uint64_t test_3 = math::lcmSum(n);
    assert(test_3 == 320);
    std::cout << "Passed Test 3!" << std::endl;

    n = 11;
    uint64_t test_4 = math::lcmSum(n);
    assert(test_4 == 616);
    std::cout << "Passed Test 4!" << std::endl;

    n = 15;
    uint64_t test_5 = math::lcmSum(n);
    assert(test_5 == 1110);
    std::cout << "Passed Test 5!" << std::endl;
}

 
int main() {
    test();   
    return 0;
}
