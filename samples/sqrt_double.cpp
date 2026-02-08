 
#include <cassert>
#include <iostream>

 
double Sqrt(double a) {
    if (a > 0 && a < 1) {
        return 1 / Sqrt(1 / a);
    }
    double l = 0, r = a;
     
    double epsilon = 1e-12;
    while (l <= r) {
        double mid = (l + r) / 2;
        if (mid * mid > a) {
            r = mid;
        } else {
            if (a - mid * mid < epsilon) {
                return mid;
            }
            l = mid;
        }
    }
    return -1;
}

 
int main() {
    double n{};
    std::cin >> n;
    assert(n >= 0);
     
    std::cout.precision(12);
    std::cout << std::fixed << Sqrt(n);
}
