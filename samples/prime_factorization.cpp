 
#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>

 
bool isprime[1000006];

 
std::vector<int> prime_numbers;

 
std::vector<std::pair<int, int>> factors;

 
void SieveOfEratosthenes(int N) {
     
    memset(isprime, true, sizeof isprime);

    for (int i = 2; i <= N; i++) {
        if (isprime[i]) {
            for (int j = 2 * i; j <= N; j += i) isprime[j] = false;
        }
    }

    for (int i = 2; i <= N; i++) {
        if (isprime[i])
            prime_numbers.push_back(i);
    }
}

 
void prime_factorization(int num) {
    int number = num;

    for (int i = 0; prime_numbers[i] <= num; i++) {
        int count = 0;

         
        if (number == 1) {
            break;
        }

        while (number % prime_numbers[i] == 0) {
            count++;
            number = number / prime_numbers[i];
        }

        if (count)
            factors.push_back(std::make_pair(prime_numbers[i], count));
    }
}

 
int main() {
    int num;
    std::cout << "\t\tComputes the prime factorization\n\n";
    std::cout << "Type in a number: ";
    std::cin >> num;

    SieveOfEratosthenes(num);

    prime_factorization(num);

     
    for (auto it : factors) {
        std::cout << it.first << " " << it.second << std::endl;
    }

    return 0;
}
