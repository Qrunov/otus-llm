 
#include <cassert>    
#include <cmath>      
#include <iostream>   

 
int number_of_digits(int num) {
    int total_digits = 0;
    while (num > 0) {
        num = num / 10;
        ++total_digits;
    }
    return total_digits;
}

 
bool is_armstrong(int number) {
     
    if (number < 0) {
        return false;
    }

    int sum = 0;
    int temp = number;
     
    int total_digits = number_of_digits(number);
    while (temp > 0) {
        int rem = temp % 10;
         
         
        sum += static_cast<int>(std::pow(rem, total_digits));
        temp = temp / 10;
    }
    return number == sum;
}

 
static void test() {
     
    assert(is_armstrong(370) == true);
     
    assert(is_armstrong(225) == false);
     
    assert(is_armstrong(-23) == false);
     
    assert(is_armstrong(153) == true);
     
    assert(is_armstrong(0) == true);
     
    assert(is_armstrong(12) == false);

    std::cout << "All tests have successfully passed!\n";
}

 
int main() {
    test();   
    return 0;
}
