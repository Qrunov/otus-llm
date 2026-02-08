 

#include <cassert>    
#include <cmath>      
#include <iostream>   

 
namespace dynamic_programming {

 
template <typename T>
bool is_armstrong(const T &number) {
    int count = 0, temp = number, result = 0, rem = 0;

     
     
    while (temp != 0) {
        temp /= 10;
        count++;
    }

     
     
     
    temp = number;
    while (temp != 0) {
        rem = temp % 10;
        result += static_cast<T>(std::pow(rem, count));
        temp /= 10;
    }

    if (result == number) {
        return true;
    } else {
        return false;
    }
}
}   

 
static void tests() {
    assert(dynamic_programming::is_armstrong(153) == true);
    assert(dynamic_programming::is_armstrong(1) == true);
    assert(dynamic_programming::is_armstrong(0) == true);
    assert(dynamic_programming::is_armstrong(370) == true);
    assert(dynamic_programming::is_armstrong(1634) == true);
    assert(dynamic_programming::is_armstrong(580) == false);
    assert(dynamic_programming::is_armstrong(15) == false);
    assert(dynamic_programming::is_armstrong(1024) == false);
    assert(dynamic_programming::is_armstrong(989) == false);
    assert(dynamic_programming::is_armstrong(103) == false);

    std::cout << "All tests have successfully passed!\n";
}

 
int main() {
    tests();   
    return 0;
}
