 
#include <cstring>
#include <iostream>

 
char prime[100000000];

 
void Sieve(int64_t n) {
    memset(prime, '1', sizeof(prime));   
    prime[0] = '0';                      
    prime[1] = '0';                      
    for (int64_t p = 2; p * p <= n; p++) {
        if (prime[p] == '1') {
            for (int64_t i = p * p; i <= n; i += p)
                prime[i] = '0';   
        }
    }
}

 
int main() {
    Sieve(100000000);
    int64_t n;
    std::cin >> n;   
    if (prime[n] == '1')
        std::cout << "YES\n";
    else
        std::cout << "NO\n";

    return 0;
}
