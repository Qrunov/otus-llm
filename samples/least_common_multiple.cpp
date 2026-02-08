 

#include <cassert>
#include <iostream>

 
unsigned int gcd(unsigned int x, unsigned int y) {
    if (x == 0) {
        return y;
    }
    if (y == 0) {
        return x;
    }
    if (x == y) {
        return x;
    }
    if (x > y) {
         

        unsigned int temp = x / y;
        return gcd(y, x - temp * y);
    }
     

    unsigned int temp = y / x;
    return gcd(x, y - temp * x);
}

 
unsigned int lcm(unsigned int x, unsigned int y) {
  return x / gcd(x, y) * y;
}

 
void tests() {
     
    assert(((void)"LCM of 5 and 10 is 10 but lcm function gives a different "
                  "result.\n",
            lcm(5, 10) == 10));
    std::cout << "First assertion passes: LCM of 5 and 10 is " << lcm(5, 10)
              << std::endl;

     
    assert(((void)"LCM of 2 and 3 is 6 but lcm function gives a different "
                  "result.\n",
            lcm(2, 3) == 6));
    std::cout << "Second assertion passes: LCM of 2 and 3 is " << lcm(2, 3)
              << std::endl;

     
     
    assert(((void)"LCM of 987654321 and 987654321 is 987654321 but lcm function"
                  " gives a different result.\n",
            lcm(987654321, 987654321) == 987654321));
    std::cout << "Third assertion passes: LCM of 987654321 and 987654321 is "
              << lcm(987654321, 987654321)
              << std::endl;
}

 
int main() {
    tests();
    return 0;
}
