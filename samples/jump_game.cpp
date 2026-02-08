 

#include <cassert>    
#include <iostream>   
#include <vector>     

 
namespace greedy_algorithms {
 
bool can_jump(const std::vector<int> &nums) {
    size_t lastPos = nums.size() - 1;
    for (size_t i = lastPos; i != static_cast<size_t>(-1); i--) {
        if (i + nums[i] >= lastPos) {
            lastPos = i;
        }
    }
    return lastPos == 0;
}
}   

 
static void test() {
    assert(greedy_algorithms::can_jump(std::vector<int>({4, 3, 1, 0, 5})));
    assert(!greedy_algorithms::can_jump(std::vector<int>({3, 2, 1, 0, 4})));
    assert(greedy_algorithms::can_jump(std::vector<int>({5, 9, 4, 7, 15, 3})));
    assert(!greedy_algorithms::can_jump(std::vector<int>({1, 0, 5, 8, 12})));
    assert(greedy_algorithms::can_jump(std::vector<int>({2, 1, 4, 7})));

    std::cout << "All tests have successfully passed!\n";
}

 
int main() {
    test();   
    return 0;
}
