 
#include <algorithm>
#include <iostream>
#include <numeric>   
#include <vector>

 
int main() {
     
    int arr[] = {10, 20, 5, 23, 42, 15};
    int n = sizeof(arr) / sizeof(arr[0]);
    std::vector<int> vect(arr, arr + n);

    std::cout << "Vector is: ";
    for (int i = 0; i < n; i++) std::cout << vect[i] << " ";

     
    std::sort(vect.begin(), vect.end());

    std::cout << "\nVector after sorting is: ";
    for (int i = 0; i < n; i++) std::cout << vect[i] << " ";

     
    std::reverse(vect.begin(), vect.end());

    std::cout << "\nVector after reversing is: ";
    for (int i = 0; i < 6; i++) std::cout << vect[i] << " ";

    std::cout << "\nMaximum element of vector is: ";
    std::cout << *max_element(vect.begin(), vect.end());

    std::cout << "\nMinimum element of vector is: ";
    std::cout << *min_element(vect.begin(), vect.end());

     
    std::cout << "\nThe summation of vector elements is: ";
    std::cout << accumulate(vect.begin(), vect.end(), 0);

    return 0;
}
