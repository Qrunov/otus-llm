 

#include <cassert>    
#include <cstdint>    
#include <iostream>   
#include <string>     

 
namespace bit_manipulation {
 
namespace even_odd {

 
        bool is_even(std::int64_t N) {
            return (N & 1) == 0 ? true : false;
        }

    }   
}   

 
static void test() {
    using bit_manipulation::even_odd::is_even;

     
    assert(is_even(0) == true);
    assert(is_even(2) == true);
    assert(is_even(100) == true);
    assert(is_even(-4) == true);
    assert(is_even(-1000) == true);

     
    assert(is_even(1) == false);
    assert(is_even(3) == false);
    assert(is_even(101) == false);
    assert(is_even(-5) == false);
    assert(is_even(-999) == false);

    std::cout << "All test cases successfully passed!" << std::endl;
}

 
int main() {
    test();   
    return 0;
}
