 
#include <cassert>
#include <iostream>

 
int sum_of_digits(int num) {
     
    if (num < 0) {
        num = -1 * num;
    }
    int sum = 0;
    while (num > 0) {
        sum = sum + (num % 10);
        num = num / 10;
    }
    return sum;
}

 
void test1() {
    int test_case_1 = sum_of_digits(119765);
    assert(test_case_1 == 29);
}

 
void test2() {
    int test_case_2 = sum_of_digits(-12256);
    assert(test_case_2 == 16);
}

 
void test() {
     
    test1();
     
    test2();
}

 
int main() {
    test();
    std::cout << "Success." << std::endl;
    return 0;
}
