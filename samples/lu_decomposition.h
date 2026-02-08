 
#pragma once

#include <iostream>
#include <valarray>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

 
template <typename T>
using matrix = std::vector<std::valarray<T>>;

 
template <typename T>
int lu_decomposition(const matrix<T> &A, matrix<double> *L, matrix<double> *U) {
    int row, col, j;
    int mat_size = A.size();

    if (mat_size != A[0].size()) {
         
        std::cerr << "Not a square matrix!\n";
        return -1;
    }

     
    for (row = 0; row < mat_size; row++) {
         
#ifdef _OPENMP
#pragma omp for
#endif
        for (col = row; col < mat_size; col++) {
             
            double lu_sum = 0.;
            for (j = 0; j < row; j++) {
                lu_sum += L[0][row][j] * U[0][j][col];
            }

             
            U[0][row][col] = A[row][col] - lu_sum;
        }

         
#ifdef _OPENMP
#pragma omp for
#endif
        for (col = row; col < mat_size; col++) {
            if (row == col) {
                L[0][row][col] = 1.;
                continue;
            }

             
            double lu_sum = 0.;
            for (j = 0; j < row; j++) {
                lu_sum += L[0][col][j] * U[0][j][row];
            }

             
            L[0][col][row] = (A[col][row] - lu_sum) / U[0][row][row];
        }
    }

    return 0;
}

 
template <typename T>
double determinant_lu(const matrix<T> &A) {
    matrix<double> L(A.size(), std::valarray<double>(A.size()));
    matrix<double> U(A.size(), std::valarray<double>(A.size()));

    if (lu_decomposition(A, &L, &U) < 0)
        return 0;

    double result = 1.f;
    for (size_t i = 0; i < A.size(); i++) {
        result *= L[i][i] * U[i][i];
    }
    return result;
}
