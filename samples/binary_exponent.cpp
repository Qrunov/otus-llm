 

#include <iostream>

 
 
int binExpo(int a, int b) {
    if (b == 0) {
        return 1;
    }
    int res = binExpo(a, b / 2);
    if (b % 2) {
        return res * res * a;
    } else {
        return res * res;
    }
}

 
 
int binExpo_alt(int a, int b) {
    int res = 1;
    while (b > 0) {
        if (b % 2) {
            res = res * a;
        }
        a = a * a;
        b /= 2;
    }
    return res;
}

 
int main() {
    int a, b;
     
    std::cin >> a >> b;
    if (a == 0 && b == 0) {
        std::cout << "Math error" << std::endl;
    } else if (b < 0) {
        std::cout << "Exponent must be positive !!" << std::endl;
    } else {
        int resRecurse = binExpo(a, b);
         

         
        std::cout << resRecurse << std::endl;
         
    }
}
