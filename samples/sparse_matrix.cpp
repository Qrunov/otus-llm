 

#include <iostream>

 
int main() {
    int m, n;
    int counterZeros = 0;

    std::cout << "Enter dimensions of matrix (seperated with space): ";
    std::cin >> m;
    std::cin >> n;

    int **a = new int *[m];
    for (int i = 0; i < m; i++) a[i] = new int[n];

    std::cout << "Enter matrix elements:";
    std::cout << "\n";

     
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << "element? ";
            std::cin >> a[i][j];
        }
    }

     
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (a[i][j] == 0)
                counterZeros++;   
        }
    }

     
    if (counterZeros > ((m * n) / 2))   
        std::cout << "Sparse matrix";
    else
        std::cout << "Not a sparse matrix";

    for (int i = 0; i < m; i++) delete[] a[i];
    delete[] a;
    return 0;
}
