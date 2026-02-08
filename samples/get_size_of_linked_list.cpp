#include <iostream>

class Node {
 public:
    int val;
    Node *next;

    Node(int v, Node *n) : val(v), next(n) {}   
};

int getSize(Node *root) {
    if (root == NULL) {
        return 0;
    }
     
    return 1 + getSize(root->next);
}

 
void deleteList(Node *const root) {
    if (root != NULL)
    {
        deleteList(root->next);
        delete root;
    }
}

int main() {
    Node *myList = new Node(0, NULL);   
    Node *temp = myList;
     
    for (int i = 1; i < 10; i++) {
        temp->next = new Node(i, NULL);
        temp = temp->next;
    }
     
    Node *secondList = new Node(0, NULL);   
    Node *thirdList = NULL;                 

    std::cout << getSize(myList) << std::endl
              << getSize(secondList) << std::endl
              << getSize(thirdList) << std::endl;
    deleteList(secondList);
    deleteList(myList);

    return 0;
}
