 
#include <cassert>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "./stack.hpp"

int main(int argc, char* argv[]) {
    double GPA = NAN;
    double highestGPA = NAN;
    std::string name;

    assert(argc == 2);
    std::ifstream infile;
    stack<std::string> stk;

    infile.open(argv[1]);
    std::cout << std::fixed << std::showpoint;
    std::cout << std::setprecision(2);
    infile >> GPA >> name;
    highestGPA = GPA;

    while (infile) {
        if (GPA > highestGPA) {
            stk.clear();
            stk.push(name);
            highestGPA = GPA;
        } else if (GPA == highestGPA) {
            stk.push(name);
        }
        infile >> GPA >> name;
    }
    std::cout << "Highest GPA: " << highestGPA << std::endl;
    std::cout << "Students the highest GPA are: " << std::endl;
    while (!stk.isEmptyStack()) {
        std::cout << stk.top() << std::endl;
        stk.pop();
    }
    std::cout << std::endl;
    return 0;
}
