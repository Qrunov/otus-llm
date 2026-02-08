 

#include <iostream>   
#include <cassert>    


 
namespace math {
 
int power_of_two(int n) {
     
     
    int result = n & (n - 1);
    
    if (result == 0) {
        return 1;
    }

    return 0;
}
}   

 
static void test() {
    std::cout << "First case testing... \n";  
    assert(math::power_of_two(32) == 1);
    std::cout << "\nPassed!\n";

    std::cout << "Second case testing... \n";  
    assert(math::power_of_two(5) == 0);
    std::cout << "\nPassed!\n";

    std::cout << "Third case testing... \n";  
    assert(math::power_of_two(232) == 0);
    std::cout << "\nPassed!\n";

    std::cout << "\nAll test cases have successfully passed!\n";
}

 
void user_input_test() {
    int n = 0;  
    
    std::cout << "Enter a number " << std::endl;
    std::cin >> n; 

     
    int result = math::power_of_two(n);
    if (result == 1) {
        std::cout << "Yes, the number " << n << " is a power of 2\n";
    }
    else { 
        std::cout << "No, the number " << n << " is not a power of 2\n";
    }
}

 
int main() {
    test();  

     
     

    return 0;
}
