 

#include <iostream>
#include <string>
#include <vector>
using namespace std;

int min(int x, int y, int z) { return min(min(x, y), z); }

 
int editDist(string str1, string str2, int m, int n) {
    if (m == 0)
        return n;
    if (n == 0)
        return m;

     
     
    if (str1[m - 1] == str2[n - 1])
        return editDist(str1, str2, m - 1, n - 1);

     
     
     
    return 1 + min(editDist(str1, str2, m, n - 1),
                   editDist(str1, str2, m - 1, n),
                   editDist(str1, str2, m - 1, n - 1));
}

 
int editDistDP(string str1, string str2, int m, int n) {
     
    std::vector<std::vector<int> > dp(m + 1, std::vector<int>(n + 1));

     
    for (int i = 0; i <= m; i++) {
        for (int j = 0; j <= n; j++) {
             
            if (i == 0)
                dp[i][j] = j;

             
            else if (j == 0)
                dp[i][j] = i;

             
            else if (str1[i - 1] == str2[j - 1])
                dp[i][j] = dp[i - 1][j - 1];

            else
                dp[i][j] = 1 + min(dp[i][j - 1],      
                                   dp[i - 1][j],      
                                   dp[i - 1][j - 1]   
                               );
        }
    }

    return dp[m][n];
}

int main() {
    string str1 = "sunday";
    string str2 = "saturday";

    cout << editDist(str1, str2, str1.length(), str2.length()) << endl;
    cout << editDistDP(str1, str2, str1.length(), str2.length()) << endl;

    return 0;
}
