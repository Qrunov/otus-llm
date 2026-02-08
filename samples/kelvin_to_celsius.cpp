 

#include <cassert>    
#include <cmath>      
#include <iostream>   

 
namespace others {
 
bool are_almost_equal(double a, double b, double absolute_tolerance = 0.0001) {
    return std::abs(a - b) < absolute_tolerance;
}

 
double kelvin_to_celsius(double temperature_in_k) {
    const double absolute_zero_in_c = -273.15;
    if (temperature_in_k < absolute_zero_in_c) {
        throw std::invalid_argument("input temperature below absolute zero");
    }
    return temperature_in_k + absolute_zero_in_c;
}
}   

 
static void tests() {
    assert(others::are_almost_equal(others::kelvin_to_celsius(230), -43.15));
    assert(others::are_almost_equal(others::kelvin_to_celsius(512), 238.85));
    assert(others::are_almost_equal(others::kelvin_to_celsius(55), -218.15));
    assert(others::are_almost_equal(others::kelvin_to_celsius(77), -196.15));
    assert(others::are_almost_equal(others::kelvin_to_celsius(9.78), -263.37));
    assert(others::are_almost_equal(others::kelvin_to_celsius(15), -258.15));
    assert(others::are_almost_equal(others::kelvin_to_celsius(273.15), 0));

    std::cout << "All tests have successfully passed!\n";
}

 
int main() {
    tests();   
    return 0;
}
