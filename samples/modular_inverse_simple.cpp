 

#include <cassert>    
#include <cstdint>
#include <iostream>   

 
uint64_t imod(uint64_t x, uint64_t y) {
    uint64_t aux = 0;   
    uint64_t itr = 0;   

    do {   
        aux = y * itr + 1;
        itr++;
    } while (aux % x);   

    return aux / x;
}

 
static void test() {
    std::cout << "First case testing... \n";
     
    assert(imod(3, 11) == 4);
    std::cout << "\nPassed!\n";

    std::cout << "Second case testing... \n";
     
    assert(imod(3, 26) == 9);
    std::cout << "\nPassed!\n";

    std::cout << "Third case testing... \n";
     
    assert(imod(7, 26) == 15);
    std::cout << "\nPassed!\n";

    std::cout << "\nAll test cases have successfully passed!\n";
}
 
int main() {
    test();   
};
