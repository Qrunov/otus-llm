 
 
 

#include <iostream>
#include <set>
#include <vector>

using namespace std;
int LIS(const std::vector<int>& arr, int n) {
    set<int> active;   
    active.insert(arr[0]);
     
    for (int i = 1; i < n; ++i) {
        auto get = active.lower_bound(arr[i]);
        if (get == active.end()) {
            active.insert(arr[i]);
        }   
        else {
            int val = *get;   
                              
            if (val > arr[i]) {
                 
                 
                active.erase(get);
                active.insert(arr[i]);
            }
        }
    }
    return active.size();   
}
int main() {
    int n;
    cout << "Enter size of array: ";
    cin >> n;
    std::vector<int> a(n);
    cout << "Enter array elements: ";
    for (int i = 0; i < n; ++i) {
        cin >> a[i];
    }
    cout << LIS(a, n) << endl;
    return 0;
}
