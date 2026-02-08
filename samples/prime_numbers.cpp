 
#include <iostream>
#include <vector>

 
std::vector<int> primes(size_t max) {
    std::vector<int> res;
    std::vector<bool> is_not_prime(max + 1, false);
    for (size_t i = 2; i <= max; i++) {
        if (!is_not_prime[i]) {
            res.emplace_back(i);
        }
        for (int p : res) {
            size_t k = i * p;
            if (k > max) {
                break;
            }
            is_not_prime[k] = true;
            if (i % p == 0) {
                break;
            }
        }
    }
    return res;
}

 
int main() {
    std::cout << "Calculate primes up to:\n>> ";
    int n = 0;
    std::cin >> n;
    std::vector<int> ans = primes(n);
    for (int p : ans) std::cout << p << ' ';
    std::cout << std::endl;
}
