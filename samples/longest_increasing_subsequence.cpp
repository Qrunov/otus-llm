 

#include <cassert>    
#include <climits>    
#include <cstdint>    
#include <iostream>   
#include <vector>     

 
namespace dynamic_programming {
 
uint64_t LIS(const std::vector<uint64_t> &a, const uint32_t &n) {
    std::vector<int> lis(n);
    for (int i = 0; i < n; ++i) {
        lis[i] = 1;
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            if (a[i] > a[j] && lis[i] < lis[j] + 1) {
                lis[i] = lis[j] + 1;
            }
        }
    }
    int res = 0;
    for (int i = 0; i < n; ++i) {
        res = std::max(res, lis[i]);
    }
    return res;
}
}   

 
static void test() {
    std::vector<uint64_t> a = {15, 21, 2, 3, 4, 5, 8, 4, 1, 1};
    uint32_t n = a.size();

    uint32_t result = dynamic_programming::LIS(a, n);
    assert(result ==
           5);   

    std::cout << "Self-test implementations passed!" << std::endl;
}

 
int main() {
    uint32_t n = 0;

    std::cout << "Enter size of array: ";
    std::cin >> n;

    std::vector<uint64_t> a(n);

    std::cout << "Enter array elements: ";
    for (int i = 0; i < n; ++i) {
        std::cin >> a[i];
    }

    std::cout << "\nThe result is: " << dynamic_programming::LIS(a, n)
              << std::endl;
    test();   

    return 0;
}
