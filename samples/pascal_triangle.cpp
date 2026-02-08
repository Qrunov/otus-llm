 
#ifdef _MSC_VER
#include <string>   
#else
#include <cstring>
#endif
#include <iomanip>
#include <iostream>

 
void show_pascal(int **arr, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n + i; ++j) {
            if (arr[i][j] == 0)
                std::cout << std::setw(4) << " ";
            else
                std::cout << std::setw(4) << arr[i][j];
        }
        std::cout << std::endl;
    }
}

 
int **pascal_triangle(int **arr, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = n - i - 1; j < n + i; ++j) {
            if (j == n - i - 1 || j == n + i - 1)
                arr[i][j] = 1;   
            else
                arr[i][j] = arr[i - 1][j - 1] + arr[i - 1][j + 1];
        }
    }

    return arr;
}

 
int main() {
    int n = 0;

    std::cout << "Set Pascal's Triangle Height" << std::endl;
    std::cin >> n;

     
    int **arr = new int *[n];
    for (int i = 0; i < n; ++i) {
        arr[i] = new int[2 * n - 1];
        memset(arr[i], 0, sizeof(int) * (2 * n - 1));
    }

    pascal_triangle(arr, n);
    show_pascal(arr, n);

     
    for (int i = 0; i < n; ++i) {
        delete[] arr[i];
    }
    delete[] arr;

    return 0;
}
