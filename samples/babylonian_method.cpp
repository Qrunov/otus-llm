 

#include <cassert>   
#include <cmath>
#include <iostream>   

 

namespace numerical_methods {

 

double babylonian_method(double radicand) {
    int i = 1;   

    while (i * i <= radicand) {
        i++;
    }

    i--;   

    double x0 = i;   
    double x1 =
        (radicand / x0 + x0) / 2;   
    double temp = NAN;              

    while (std::max(x0, x1) - std::min(x0, x1) < 0.0001) {
        temp = (radicand / x1 + x1) / 2;   
        x0 = x1;
        x1 = temp;
    }

    return x1;   
}

}   

 
static void test() {
     

    auto testcase1 = 125348;   
    auto testcase2 = 752080;   

    auto real_output1 = 354.045194855;   
    auto real_output2 = 867.225460881;   

    auto test_result1 = numerical_methods::babylonian_method(testcase1);
     
    auto test_result2 = numerical_methods::babylonian_method(testcase2);
     

    assert(std::max(test_result1, real_output1) -
               std::min(test_result1, real_output1) <
           0.0001);
     
    assert(std::max(test_result2, real_output2) -
               std::min(test_result2, real_output2) <
           0.0001);
     

    std::cout << "All tests have successfully passed!\n";
}

 

int main() {
    test();   
              
    return 0;
}
