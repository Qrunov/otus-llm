 

#include <cassert>    
#include <climits>    
#include <iostream>   
#include <vector>     

 
namespace dynamic_programming {
 
namespace mincoins_topdown {
 
template <typename T>
int64_t mincoins(const T &n, const std::vector<T> &coins, const int16_t &t,
                 std::vector<T> dp) {
    if (n == 0) {
        return 0;
    }
    if (dp[n] != 0) {
        return dp[n];
    }
    int ans = INT_MAX;   
    for (int i = 0; i < t; i++) {
        if (n - coins[i] >= 0) {   
                                   
            int sub = mincoins(n - coins[i], coins, t, dp);
            ans = std::min(ans, sub + 1);
        }
    }
    dp[n] = ans;
    return dp[n];   
}

}   
}   

 
static void test() {
     
    const int64_t n1 = 15;
    const int8_t t1 = 3, a1 = 0;
    std::cout << "\nTest 1...";
    std::vector<int64_t> arr1{1, 7, 10};
    std::vector<int64_t> dp1(n1 + 1);
    fill(dp1.begin(), dp1.end(), a1);
    assert(dynamic_programming::mincoins_topdown::mincoins(n1, arr1, t1, dp1) ==
           3);
    std::cout << "Passed\n";
}

 
int main() {
    test();   
    return 0;
}
