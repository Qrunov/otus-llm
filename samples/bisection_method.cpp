 
#include <cmath>
#include <iostream>
#include <limits>

#define EPSILON \
    1e-6   
#define MAX_ITERATIONS 50000   

 
static double eq(double i) {
    return (std::pow(i, 3) - (4 * i) - 9);   
}

 
template <typename T>
int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

 
int main() {
    double a = -1, b = 1, x, z;
    int i;

     
    for (int i = 0; i < MAX_ITERATIONS; i++) {
        z = eq(a);
        x = eq(b);
        if (sgn(z) == sgn(x)) {   
            b++;
            a--;
        } else {   
            break;
        }
    }

    std::cout << "\nFirst initial: " << a;
    std::cout << "\nSecond initial: " << b;

     
    for (i = 0; i < MAX_ITERATIONS; i++) {
        x = (a + b) / 2;
        z = eq(x);
        std::cout << "\n\nz: " << z << "\t[" << a << " , " << b
                  << " | Bisect: " << x << "]";

        if (z < 0) {
            a = x;
        } else {
            b = x;
        }

        if (std::abs(z) < EPSILON)   
            break;
    }

    std::cout << "\n\nRoot: " << x << "\t\tSteps: " << i << std::endl;
    return 0;
}
