#include <climits>
#include <iostream>
using namespace std;

#define MAX 10

 
int dp[MAX][MAX];

 
 
int MatrixChainMultiplication(int dim[], int i, int j) {
     
    if (j <= i + 1)
        return 0;

     
     
    int min = INT_MAX;

     

    if (dp[i][j] == 0) {
         
         

        for (int k = i + 1; k <= j - 1; k++) {
             
            int cost = MatrixChainMultiplication(dim, i, k);

             
            cost += MatrixChainMultiplication(dim, k, j);

             
            cost += dim[i] * dim[k] * dim[j];

            if (cost < min)
                min = cost;   
        }
        dp[i][j] = min;
    }

     
    return dp[i][j];
}

 
int main() {
     
     
    int dim[] = {10, 30, 5, 60};
    int n = sizeof(dim) / sizeof(dim[0]);

     
     

    cout << "Minimum cost is " << MatrixChainMultiplication(dim, 0, n - 1)
         << "\n";

    return 0;
}