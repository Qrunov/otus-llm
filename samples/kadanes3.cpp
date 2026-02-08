 

#include <array>      
#include <cassert>    
#include <climits>    
#include <cstdint>
#include <iostream>   
 
template <std::size_t SIZE>
int max_subarray_sum(std::array<int64_t, SIZE> arr, uint64_t length) {
    int64_t current_max = INT_MIN, current_sum = 0;
    for (int i = 0; i < length; i++) {
        current_sum = current_sum + arr[i];
        if (current_max < current_sum) {
            current_max = current_sum;
        }

        if (current_sum < 0) {
            current_sum = 0;
        }
    }
    return current_max;
}

 
static void test() {
    std::array<int64_t, 4> arr = {1, 2, 3, 4};
    std::array<int64_t, 5> arr1 = {-1, -2, -4, -6, 7};
    assert(max_subarray_sum(arr, 4) == 10);
    assert(max_subarray_sum(arr1, 5) == 7);
    std::cout << "All test cases have passed!\n";
}

 
int main() {
     
     
     
     
     

     
     
     
     
     
     
     
     
     
     
     
     

     
     
     

    test();   
    return 0;
}
