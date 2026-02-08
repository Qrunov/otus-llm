 
#include <algorithm>   
#include <iostream>
#include <cstdint>

 
template <typename T, typename T2>
inline void update_step(T *r, T *r0, const T2 quotient) {
    T temp = *r;
    *r = *r0 - (quotient * temp);
    *r0 = temp;
}

 
template <typename T1, typename T2>
void extendedEuclid_1(T1 A, T1 B, T1 *GCD, T2 *x, T2 *y) {
    if (B > A)
        std::swap(A, B);   

    T2 s = 0, s0 = 1;
    T2 t = 1, t0 = 0;
    T1 r = B, r0 = A;

    while (r != 0) {
        T1 quotient = r0 / r;
        update_step(&r, &r0, quotient);
        update_step(&s, &s0, quotient);
        update_step(&t, &t0, quotient);
    }
    *GCD = r0;
    *x = s0;
    *y = t0;
}

 
template <typename T, typename T2>
void extendedEuclid(T A, T B, T *GCD, T2 *x, T2 *y) {
    if (B > A)
        std::swap(A, B);   

    if (B == 0) {
        *GCD = A;
        *x = 1;
        *y = 0;
    } else {
        extendedEuclid(B, A % B, GCD, x, y);
        T2 temp = *x;
        *x = *y;
        *y = temp - (A / B) * (*y);
    }
}

 
int main() {
    uint32_t a, b, gcd;
    int32_t x, y;
    std::cin >> a >> b;
    extendedEuclid(a, b, &gcd, &x, &y);
    std::cout << gcd << " " << x << " " << y << std::endl;
    extendedEuclid_1(a, b, &gcd, &x, &y);
    std::cout << gcd << " " << x << " " << y << std::endl;
    return 0;
}
