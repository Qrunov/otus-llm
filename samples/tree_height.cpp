 

 

#include <iostream>
#include <vector>

 
 
const int MAX = 1e5;
 
std::vector<int> adj[MAX];
std::vector<bool> visited;
std::vector<int> dp;

void depth_first_search(int u) {
    visited[u] = true;
    int child_height = 1;
    for (int v : adj[u]) {
        if (!visited[v]) {
            depth_first_search(v);

             
            child_height = std::max(child_height, dp[v] + 1);
        }
    }
     
    dp[u] = child_height;
}

int main() {
     
    int number_of_nodes;
    std::cout << "Enter number of nodes of the tree : " << std::endl;
    std::cin >> number_of_nodes;

     
    int u, v;
     
    std::cout << "Enter edges of the tree : " << std::endl;
    for (int i = 0; i < number_of_nodes - 1; i++) {
        std::cin >> u >> v;
         
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
     
    visited.assign(number_of_nodes + 1, false);
     
    dp.assign(number_of_nodes + 1, 0);
     
    depth_first_search(1);
    std::cout << "Height of the Tree : " << dp[1] << std::endl;
}
