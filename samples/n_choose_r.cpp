 

#include <cassert>    
#include <cstdint>
#include <iostream>   
 
namespace math {
 
template <class T>
T n_choose_r(T n, T r) {
    if (r > n / 2) {
        r = n - r;   
    }
    T ans = 1;
    for (int i = 1; i <= r; i++) {
        ans *= n - r + i;
        ans /= i;
    }
    return ans;
}
}   

 
static void test() {
     
    uint8_t t = math::n_choose_r(5, 2);
    assert(((void)"10 is the answer but function says otherwise.\n", t == 10));
    std::cout << "First test passes." << std::endl;
     
    t = math::n_choose_r(5, 3);
    assert(
        ((void)"10 is the answer but the function says otherwise.\n", t == 10));
    std::cout << "Second test passes." << std::endl;
     
    t = math::n_choose_r(3, 2);
    assert(
        ((void)"3 is the answer but the function says otherwise.\n", t == 3));
    std::cout << "Third test passes." << std::endl;
     
    t = math::n_choose_r(10, 4);
    assert(((void)"210 is the answer but the function says otherwise.\n",
            t == 210));
    std::cout << "Fourth test passes." << std::endl;
}

 
int main() {
    test();   
    return 0;
}
