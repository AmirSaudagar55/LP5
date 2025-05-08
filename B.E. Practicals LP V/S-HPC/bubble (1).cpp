#include <iostream>
#include <vector>
#include <omp.h>
#include<ctime>
using namespace std;

void bubbleSort(vector<int>& arr) {
    int n = arr.size();
    bool swapped = true;

    while (swapped) {
        swapped = false;
        #pragma omp parallel for
        for (int i = 0; i < n - 1; i++) {
            if (arr[i] > arr[i + 1]) {
                swap(arr[i], arr[i + 1]);
                swapped = true;
            }
        }
    }
}
int main() {
    int n;
    cout << "Enter the number of elements: ";
    cin >> n;

    vector<int> arr(n);
    cout << "Enter the elements: ";
    for (int i = 0; i < n; i++)
        cin >> arr[i];

    clock_t bubbleStart = clock();
    bubbleSort(arr);
    clock_t bubbleEnd = clock();
    cout << "Sorted array using Bubble Sort: ";
    for (int num : arr)
        cout << num << " ";
    cout << endl;
    double bubbleDuration = double(bubbleEnd-bubbleStart) / CLOCKS_PER_SEC;
    cout << "Bubble sort time in seconds: " << bubbleDuration << endl;
    return 0;
}