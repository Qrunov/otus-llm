#include <climits>
#include <iostream>
#include <vector>
using namespace std;

 
int findMinCoins(int arr[], int n, int N) {
     
    std::vector<int> dp(N + 1);

     

    dp[0] = 0;

    for (int i = 1; i <= N; i++) {
         
        dp[i] = INT_MAX;
        int res = INT_MAX;

         
        for (int c = 0; c < n; c++) {
            if (i - arr[c] >=
                0)   
                res = dp[i - arr[c]];

             
             
            if (res != INT_MAX)
                dp[i] = min(dp[i], res + 1);
        }
    }

     
    return dp[N];
}

int main() {
     
    int arr[] = {1, 2, 3, 4};
    int n = sizeof(arr) / sizeof(arr[0]);

     
    int N = 15;

    cout << "Minimum Number of Coins Required " << findMinCoins(arr, n, N)
         << "\n";

    return 0;
}