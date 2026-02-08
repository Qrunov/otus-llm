 
#include <iostream>

 
struct tower {
     
    int values[10];
     
    int top;
};

 
void show(const struct tower *const F, const struct tower *const T,
          const struct tower *const U) {
    std::cout << "\n\n\tF : ";
    for (int i = 0; i < F->top; i++) {
        std::cout << F->values[i] << "\t";
    }
    std::cout << "\n\tU : ";
    for (int i = 0; i < U->top; i++) {
        std::cout << U->values[i] << "\t";
    }
    std::cout << "\n\tT : ";
    for (int i = 0; i < T->top; i++) {
        std::cout << T->values[i] << "\t";
    }
}

 
void mov(tower *From, tower *To) {
    --From->top;
    To->values[To->top] = From->values[From->top];
    ++To->top;
}

 
void TH(int n, tower *From, tower *Using, tower *To) {
    if (n == 1) {
        mov(From, To);
        show(From, To, Using);
    } else {
        TH(n - 1, From, To, Using);
        mov(From, To);
        show(From, To, Using);
        TH(n - 1, Using, From, To);
    }
}

 
int main() {
    struct tower F, U, T;

    F.top = 0;
    U.top = 0;
    T.top = 0;

    int no;

    std::cout << "\nEnter number of discs : ";
    std::cin >> no;

    for (int i = no; i > 0; i--) {
        F.values[F.top++] = i;
    }

    show(&F, &T, &U);
    TH(no, &F, &U, &T);

    return 0;
}
