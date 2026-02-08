 

#include <cassert>   
#include <cstdint>   
#include <vector>    

class MemorisedFactorial {
    std::vector<std::uint64_t> known_values = {1};

 public:
     
    std::uint64_t operator()(std::uint64_t n) {
        if (n >= this->known_values.size()) {
            this->known_values.push_back(n * this->operator()(n - 1));
        }
        return this->known_values.at(n);
    }
};

void test_MemorisedFactorial_in_order() {
    auto factorial = MemorisedFactorial();
    assert(factorial(0) == 1);
    assert(factorial(1) == 1);
    assert(factorial(5) == 120);
    assert(factorial(10) == 3628800);
}

void test_MemorisedFactorial_no_order() {
    auto factorial = MemorisedFactorial();
    assert(factorial(10) == 3628800);
}

 
int main() {
    test_MemorisedFactorial_in_order();
    test_MemorisedFactorial_no_order();
    return 0;
}
