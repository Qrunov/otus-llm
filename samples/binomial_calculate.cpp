 

#include <cassert>    
#include <cstdint>    
#include <cstdlib>    
#include <iostream>   

 
namespace math {
 
namespace binomial {
 
size_t calculate(int32_t n, int32_t k) {
     
    if (k > (n / 2))
        k = n - k;
    if (k == 1)
        return n;
    if (k == 0)
        return 1;

    size_t result = 1;
    for (int32_t i = 1; i <= k; ++i) {
        result *= n - k + i;
        result /= i;
    }

    return result;
}
}   
}   

 
static void tests() {
     
    assert(math::binomial::calculate(1, 1) == 1);
    assert(math::binomial::calculate(57, 57) == 1);
    assert(math::binomial::calculate(6, 3) == 20);
    assert(math::binomial::calculate(10, 5) == 252);
    assert(math::binomial::calculate(20, 10) == 184756);
    assert(math::binomial::calculate(30, 15) == 155117520);
    assert(math::binomial::calculate(40, 20) == 137846528820);
    assert(math::binomial::calculate(50, 25) == 126410606437752);
    assert(math::binomial::calculate(60, 30) == 118264581564861424);
    assert(math::binomial::calculate(62, 31) == 465428353255261088);

    std::cout << "[+] Binomial coefficients calculate test completed"
              << std::endl;
}

 
int main(int argc, const char* argv[]) {
    tests();   

    if (argc < 3) {
        std::cout << "Usage ./binomial_calculate {n} {k}" << std::endl;
        return 0;
    }

    int32_t n = atoi(argv[1]);
    int32_t k = atoi(argv[2]);

    std::cout << math::binomial::calculate(n, k) << std::endl;
    return 0;
}
