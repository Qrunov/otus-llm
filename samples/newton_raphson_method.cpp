 
#include <cmath>
#include <cstdint>
#include <ctime>
#include <iostream>
#include <limits>

constexpr double EPSILON = 1e-10;               
constexpr int16_t MAX_ITERATIONS = INT16_MAX;   

 
static double eq(double i) {
    return (std::pow(i, 3) - (4 * i) - 9);   
}

 
static double eq_der(double i) {
    return ((3 * std::pow(i, 2)) - 4);   
}

 
int main() {
    std::srand(std::time(nullptr));   

    double z = NAN, c = std::rand() % 100, m = NAN, n = NAN;
    int i = 0;

    std::cout << "\nInitial approximation: " << c;

     
    for (i = 0; i < MAX_ITERATIONS; i++) {
        m = eq(c);
        n = eq_der(c);

        z = c - (m / n);
        c = z;

        if (std::abs(m) < EPSILON) {   
            break;
        }
    }

    std::cout << "\n\nRoot: " << z << "\t\tSteps: " << i << std::endl;
    return 0;
}
