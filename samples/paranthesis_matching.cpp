 
#include <iostream>
#ifdef _MSC_VER
#include <string>   
#else
#include <cstring>
#endif

 
#define MAX 100

 
 
char stack[MAX];

 
int stack_idx = -1;

 
void push(char ch) { stack[++stack_idx] = ch; }

 
char pop() { return stack[stack_idx--]; }

 

 
char opening(char ch) {
    switch (ch) {
    case '}':
        return '{';
    case ']':
        return '[';
    case ')':
        return '(';
    case '>':
        return '<';
    }
    return '\0';
}

int main() {
    std::string exp;
    int valid = 1, i = 0;
    std::cout << "Enter The Expression : ";
    std::cin >> exp;

    while (valid == 1 && i < exp.length()) {
        if (exp[i] == '(' || exp[i] == '{' || exp[i] == '[' || exp[i] == '<') {
            push(exp[i]);
        } else if (stack_idx >= 0 && stack[stack_idx] == opening(exp[i])) {
            pop();
        } else {
            valid = 0;
        }
        i++;
    }

     
    if (valid == 1 && stack_idx == -1) {
        std::cout << "\nCorrect Expression";
    } else {
        std::cout << "\nWrong Expression";
    }

    return 0;
}
