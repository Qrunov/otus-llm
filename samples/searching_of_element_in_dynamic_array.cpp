 
#include <iostream>

 
 
int main() {
    int64_t r, mr = 0, x, q, i, z;
    std::cout << "Enter Number of array you want to Store :";
    std::cin >> x;
    std::cout << "Enter Number of ";
    std::cout << "Question or Quary you ";
    std::cout << "want to do Related to Array :";
    std::cin >> q;
     
     
     
    int** ar = new int*[x]();
     
     
    for (r = 0; r < x; r++) {
        std::cout << "Enter number of element in " << r + 1 << " rows :";
        std::cin >> mr;
         
        int* ac = new int[mr]();
        std::cout << "Enter the element of Array ";
         
         
        for (i = 0; i < mr; i++) {
             
            std::cin >> ac[i];
        }
         
        ar[r] = ac;
    }
     
     
    for (z = 0; z < q; z++) {
        int64_t r1 = 0, q1 = 0;
        std::cout << "enter the number of row which element you want to find :";
        std::cin >> r1;
        r1 = r1 - 1;
        std::cout << "enter the position of element which you want to find :";
        std::cin >> q1;
        q1 = q1 - 1;
         
        std::cout << "The element is " << ar[r1][q1] << std::endl;
    }
}
