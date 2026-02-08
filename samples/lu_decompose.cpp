 
#include <cassert>
#include <ctime>
#include <iomanip>
#include <iostream>

#include "./lu_decomposition.h"

 
template <typename T>
std::ostream &operator<<(std::ostream &out, matrix<T> const &v) {
    const int width = 10;
    const char separator = ' ';

    for (size_t row = 0; row < v.size(); row++) {
        for (size_t col = 0; col < v[row].size(); col++)
            out << std::left << std::setw(width) << std::setfill(separator)
                << v[row][col];
        out << std::endl;
    }

    return out;
}

 
void test1() {
    int mat_size = 3;   
    const int range = 50;
    const int range2 = range >> 1;

     
    matrix<double> A(mat_size, std::valarray<double>(mat_size));
    matrix<double> L(mat_size, std::valarray<double>(mat_size));   
    matrix<double> U(mat_size, std::valarray<double>(mat_size));   
    for (int i = 0; i < mat_size; i++) {
         
        for (int j = 0; j < mat_size; j++)
             
            A[i][j] = static_cast<double>(std::rand() % range - range2);
    }

    std::clock_t start_t = std::clock();
    lu_decomposition(A, &L, &U);
    std::clock_t end_t = std::clock();
    std::cout << "Time taken: "
              << static_cast<double>(end_t - start_t) / CLOCKS_PER_SEC << "\n";

    std::cout << "A = \n" << A << "\n";
    std::cout << "L = \n" << L << "\n";
    std::cout << "U = \n" << U << "\n";
}

 
void test2() {
    std::cout << "Determinant test 1...";
    matrix<int> A1({{1, 2, 3}, {4, 9, 6}, {7, 8, 9}});
    assert(determinant_lu(A1) == -48);
    std::cout << "passed\n";

    std::cout << "Determinant test 2...";
    matrix<int> A2({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
    assert(determinant_lu(A2) == 0);
    std::cout << "passed\n";

    std::cout << "Determinant test 3...";
    matrix<float> A3({{1.2, 2.3, 3.4}, {4.5, 5.6, 6.7}, {7.8, 8.9, 9.0}});
    assert(determinant_lu(A3) == 3.63);
    std::cout << "passed\n";
}

 
int main() {
    std::srand(std::time(NULL));   

    test1();
    test2();
    return 0;
}
