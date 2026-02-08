 
#include <cassert>    
#include <iostream>   

 
namespace math {
 
int sum_of_divisor(int num) {
     
    int sum = 1;
     
     
    for (int div = 2; div * div <= num; ++div) {
         
        if (num % div == 0) {
             
            if (div == (num / div)) {
                sum += div;
            } else {
                 
                sum += (div + (num / div));
            }
        }
    }
    return sum;
}

 
bool are_amicable(int x, int y) {
    return (sum_of_divisor(x) == y) && (sum_of_divisor(y) == x);
}
}   

 
static void tests() {
    assert(math::are_amicable(220, 284) == true);
    assert(math::are_amicable(6368, 6232) == true);
    assert(math::are_amicable(458, 232) == false);
    assert(math::are_amicable(17296, 18416) == true);
    assert(math::are_amicable(18416, 17296) == true);

    std::cout << "All tests have successfully passed!" << std::endl;
}

 
int main() {
    tests();   
    return 0;
}
