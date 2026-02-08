 
#include <array>
#include <climits>
#include <iostream>
 
namespace dynamic_programming {
 
namespace kadane {
 
template <size_t N>
int maxSubArray(const std::array<int, N> &n) {
    int curr_sum =
        0;   
    int max_sum = INT_MIN;   
    for (int i : n) {   
        curr_sum += n[i];
        max_sum = std::max(max_sum, curr_sum);   
        curr_sum = std::max(curr_sum, 0);   
    }
    return max_sum;   
}
}   
}   

 
int main() {
    const int N = 5;
    std::array<int, N> n{};   
                              
    for (int i = 0; i < n.size(); i++) {
        std::cout << "Enter value of n[" << i << "]"
                  << "\n";
        std::cin >> n[i];
    }
    int max_sum = dynamic_programming::kadane::maxSubArray<N>(
        n);   
    std::cout << "Maximum subarray sum is " << max_sum;   

    return 0;
}
