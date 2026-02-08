 

#include <cassert>    
#include <cstdlib>    
#include <iostream>   
#include <vector>     

 
namespace math {

 
using Point = struct {
    double x;
    double y;
};

 
double approximate_pi(const std::vector<Point> &pts) {
    double count = 0;   
    for (Point p : pts) {
        if ((p.x * p.x) + (p.y * p.y) <= 1) {
            count++;
        }
    }
    return 4.0 * count / static_cast<double>(pts.size());
}
}   

 
static void tests() {
    std::vector<math::Point> rands;
    for (std::size_t i = 0; i < 100000; i++) {
        math::Point p;
        p.x = rand() / static_cast<double>(RAND_MAX);   
        p.y = rand() / static_cast<double>(RAND_MAX);   
        rands.push_back(p);
    }
    assert(math::approximate_pi(rands) > 3.135);
    assert(math::approximate_pi(rands) < 3.145);

    std::cout << "All tests have successfully passed!" << std::endl;
}

 
int main() {
    tests();   
    return 0;
}
