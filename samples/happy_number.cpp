 

#include <iostream>

 
template <typename T>
bool is_happy(T n) {
    T s = 0;               
    while (n > 9) {        
        while (n != 0) {   
            T d = n % 10;
            s += d;
            n /= 10;
        }
        n = s;
        s = 0;
    }
    return (n == 1) ? true : false;   
}

 
int main() {
    int n;
    std::cout << "Enter a number:";
    std::cin >> n;

    if (is_happy(n))
        std::cout << n << " is a happy number" << std::endl;
    else
        std::cout << n << " is not a happy number" << std::endl;
    return 0;
}
