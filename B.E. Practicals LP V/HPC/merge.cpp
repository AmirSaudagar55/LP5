#include <iostream>
#include <vector>
#include <omp.h>
#include<ctime>

using namespace std;

void merge(vector<int>& arr, int l, int m, int r) {
    vector<int> temp;
    int left = l;
    int right = m + 1;

    while (left <= m && right <= r) {
        if (arr[left] <= arr[right]) {
            temp.push_back(arr[left]);
            left++;
        }
        else {
            temp.push_back(arr[right]);
            right++;
        }
    }

    while (left <= m) {
        temp.push_back(arr[left]);
        left++;
    }

    while (right <= r) {
        temp.push_back(arr[right]);
        right++;
    }

    for (int i = l; i <= r; i++) {
        arr[i] = temp[i - l];
    }
}

void mergeSort(vector<int>& arr, int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;
        #pragma omp parallel sections
        {
            #pragma omp section
            mergeSort(arr, l, m);
            #pragma omp section
            mergeSort(arr, m + 1, r);
        }
        merge(arr, l, m, r);
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
        clock_t mergeStart = clock();
    mergeSort(arr, 0, n - 1);
    clock_t mergeEnd = clock();
    cout << "Sorted array using Merge Sort: ";
    for (int num : arr)
        cout << num << " ";
    cout << endl;

    double mergeDuration = double(mergeEnd-mergeStart) / CLOCKS_PER_SEC;
    cout << "Merge sort time in seconds: " << mergeDuration << endl;

    return 0;
}














































































Thought for a couple of seconds


Here’s what your OpenMP‑powered Merge Sort is doing, line by line, and a tiny worked‑through example so you can see the parallelism in action.

---

## 1. The Key Pragmas

```cpp
#pragma omp parallel sections
{
    #pragma omp section
    mergeSort(arr, l, m);
    #pragma omp section
    mergeSort(arr, m+1, r);
}
```

1. **`#pragma omp parallel sections`**

   * **Spawns a team of threads** (by default, one per available core).
   * Inside this block, each `#pragma omp section` is handed off to exactly one of those threads.
   * When all sections are dispatched, threads implicitly synchronize (barrier) at the end of the `parallel sections` block before proceeding to the `merge(...)` call.

2. **`#pragma omp section`**

   * Marks a chunk of code (here, a recursive call to sort one half of the array) that exactly one thread will execute.
   * With two sections, you get up to two threads working concurrently—one sorting the left half, one sorting the right half.

---

## 2. Why Use Parallel Sections Here?

Merge Sort naturally splits the problem in two independent halves:

```
mergeSort(arr, l, m);      // sort left half
mergeSort(arr, m+1, r);    // sort right half
merge(arr, l, m, r);       // then merge them
```

By doing the two recursive `mergeSort` calls in parallel, you can utilize two cores (or threads) to halve the wall‑clock time of the divide-and-conquer step—ideally close to a 2× speedup for large subarrays.

---

## 3. Worked‑Through Example

Let’s sort the array `[5, 2, 8, 1]`. We call:

```cpp
mergeSort(arr, 0, 3);
```

So `l=0, r=3, m = 1`.

### Step A: Parallel Sections at the Top Level

```cpp
#pragma omp parallel sections
{
  #pragma omp section        // Thread 1
      mergeSort(arr, 0, 1);   // sorts [5,2]
  #pragma omp section        // Thread 2
      mergeSort(arr, 2, 3);   // sorts [8,1]
}
// implicit barrier: wait for both threads to finish
merge(arr, 0, 1, 3);
```

* **Thread 1** works on indices `0…1`:

  * Computes `m = 0`, then (serially, since no further pragmas inside small ranges) sorts `[5]` and `[2]` and merges them → `[2,5]`.
* **Thread 2** works on indices `2…3`:

  * Computes `m = 2`, sorts `[8]` and `[1]`, merges → `[1,8]`.

Once **both threads** hit the end of their `sections` block, they join, and the master thread calls:

```cpp
merge(arr, 0, 1, 3);
```

to merge the two sorted halves `[2,5]` and `[1,8]` into the final sorted array `[1,2,5,8]`.

---

## 4. Important Caveats & Tips

* **Thread Overhead**
  Each `parallel sections` creates and tears down a thread team. In a deep recursion, that overhead can dominate for small subarrays. You can mitigate this by only spawning new sections when the subarray length exceeds a threshold:

  ```cpp
  if (r - l + 1 > PARALLEL_THRESHOLD) {
    #pragma omp parallel sections
    { ... }
  } else {
    mergeSort(arr, l, m);
    mergeSort(arr, m+1, r);
  }
  ```

* **Shared Array Safety**
  All threads share the same `arr` memory, but they only work on **disjoint** index ranges (`l…m` vs. `m+1…r`), so there’s no race condition during the recursive sorts. The `merge(...)` happens only after the barrier, when both halves are fully sorted.

* **Load Balancing**
  Ideal speedup comes if both halves take roughly equal time. In very skewed splits (e.g. always splitting into sizes 1 and N–1), one thread will sit idle while the other does most of the work.

---

### In Practice

On a 4‑core machine, you might see:

```
Enter the number of elements: 8
Enter the elements: 5 2 8 1 7 3 6 4
Sorted array using Merge Sort: 1 2 3 4 5 6 7 8
Merge sort time in seconds: 0.000023
```

Behind the scenes, at the top split you got 2× parallelism; at the next level of recursion each half could itself spawn two threads, up to the max thread count. With a large enough array and a reasonable threshold, you could effectively use all 4 or 8 cores to cut runtime by nearly that factor.

---

**Summary:**

* **`parallel sections`** divides distinct code blocks among threads.
* Each **`section`** runs in its own thread, then all threads sync before merging.
* It leverages Merge Sort’s natural divide‑and‑conquer to exploit parallel hardware with minimal code changes.
