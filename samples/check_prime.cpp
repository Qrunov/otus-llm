 

#include <cassert>    
#include <iostream>   

 
namespace math {
 
bool is_prime(int64_t num) {
     
    if (num <= 1) {
        return false;
    } else if (num == 2 || num == 3) {
        return true;
    } else if (num % 2 == 0 || num % 3 == 0) {
        return false;
    } else {
        for (int64_t i = 5; i * i <= num; i = i + 6) {
            if (num % i == 0 || num % (i + 2) == 0) {
                return false;
            }
        }
    }
    return true;
}
}   

 
static void tests() {
    assert(math::is_prime(1) == false);
    assert(math::is_prime(2) == true);
    assert(math::is_prime(3) == true);
    assert(math::is_prime(4) == false);
    assert(math::is_prime(-4) == false);
    assert(math::is_prime(7) == true);
    assert(math::is_prime(-7) == false);
    assert(math::is_prime(19) == true);
    assert(math::is_prime(50) == false);
    assert(math::is_prime(115249) == true);

    std::cout << "All tests have successfully passed!" << std::endl;
}

 
int main() {
    tests();   
    return 0;
}
