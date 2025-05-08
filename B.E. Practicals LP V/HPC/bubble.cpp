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



























































Thought for a few seconds


Here’s what your OpenMP‑augmented Bubble Sort is doing, and a tiny worked‑through example so you can see it in action.

---

## 1. The OpenMP Directive

```cpp
#pragma omp parallel for
for (int i = 0; i < n - 1; i++) {
    if (arr[i] > arr[i + 1]) {
        swap(arr[i], arr[i + 1]);
        swapped = true;
    }
}
```

1. **`#pragma omp parallel for`**

   * **Spawns a team of threads** and splits the iterations of the immediately following `for` loop across them.
   * Each thread gets its own private copy of the loop index `i`, and works on a distinct subset of `i` values simultaneously.

2. **Variable scoping**

   * By default under OpenMP, variables declared **outside** the parallel region (like `arr`, `n`, and `swapped`) are **shared** among threads.
   * The loop index `i` is automatically made **private** (each thread has its own `i`).
   * **Caution:** `swapped` is shared and being written to by multiple threads without synchronization—this can lead to a data race on that flag. Despite that, setting `swapped = true` by any thread “works” in practice to force another pass, but it’s technically undefined behavior unless you protect it (e.g. with `#pragma omp atomic`).

3. **Implicit barrier**

   * At the end of the `parallel for`, there is an **implicit barrier**: all threads wait until every iteration is done before any thread proceeds to the next statement (the `while(swapped)` check).

4. **Why (attempt to) parallelize Bubble Sort?**

   * Bubble Sort’s inner loop compares and swaps adjacent elements. In a **standard**, **serial** Bubble Sort, you must do these comparisons in strict sequence to guarantee correctness.
   * By putting the compare‑swap across the entire array in a single parallel loop, you get a form of **odd–even transposition sort**, where in one pass you compare elements `(0,1),(2,3),(4,5)…` in parallel, and on the next pass you compare `(1,2),(3,4),(5,6)…`.
   * However, your code compares **all** adjacent pairs every pass, which can still converge but may swap the “wrong” neighboring pairs simultaneously—so be aware that a naïve parallel for on bubble sort can give incorrect results unless the algorithm is adjusted.

---

## 2. Step‑by‑Step Example

Let’s run through one `while(swapped)` pass on a small array of size 6:

```
Initial arr: [ 6, 3, 5, 2, 4, 1 ]
 n = 6, so i runs from 0…4
```

### Parallel For Iterations

* **Thread A**: handles i = 0 → compares (6,3) → swap → arr becomes `[3,6,5,2,4,1]`, sets `swapped=true`
* **Thread B**: handles i = 1 → compares (6,5) → swap → arr becomes `[3,5,6,2,4,1]`
* **Thread C**: handles i = 2 → compares (6,2) → swap → arr becomes `[3,5,2,6,4,1]`
* **Thread D**: handles i = 3 → compares (6,4) → swap → arr becomes `[3,5,2,4,6,1]`
* **Thread E**: handles i = 4 → compares (6,1) → swap → arr becomes `[3,5,2,4,1,6]`

*(iteration order can interleave, but all adjacent swaps happen “simultaneously”)*

After this parallel loop completes (all threads hit the barrier), you’ve effectively bubbled the largest element `6` to the end—just like one full pass of serial Bubble Sort.

Because `swapped` was set to true by at least one thread, the `while(swapped)` loop repeats. On the next pass you’ll move the second‑largest element (`5`) into position 4, and so on, until the array is fully sorted.

---

## 3. Caveats & Best Practices

* **Data Races on `swapped`**

  * Multiple threads write `swapped = true` without coordination. To make it safe, you could do:

    ```cpp
    #pragma omp atomic write
    swapped = true;
    ```

    or use a reduction clause to OR flags, but that’s more boilerplate.

* **Algorithmic Suitability**

  * Bubble Sort isn’t a naturally parallel algorithm—adjacent‐swap dependencies mean you usually need an odd–even transposition variant to get a correct parallel sort.
  * For real‑world parallel sorting, you’d pick something like parallel quicksort, samplesort, or rely on library routines (e.g. Intel TBB’s `parallel_sort`).

---

### In Practice

If you run the above code on a 4‑core CPU with our 6‑element example, you’ll likely see something like:

```
Enter the number of elements: 6
Enter the elements: 6 3 5 2 4 1
Sorted array using Bubble Sort: 1 2 3 4 5 6 
Bubble sort time in seconds: 0.000012
```

Behind the scenes, each pass through the array was divided among your CPU’s threads, each swapping in parallel. For small `n` the overhead outweighs any speedup, but on very large arrays (and with an odd–even parallel sort variant), you can potentially reduce wall‑clock time by a factor close to your core count.

---

**Summary:**

* `#pragma omp parallel for` splits a loop across threads.
* You must mind **shared** vs. **private** variables and guard any writes (like to `swapped`) if you want fully defined behavior.
* Bubble Sort needs algorithmic tweaks (odd–even passes) to correctly exploit parallelism—straight parallelizing the inner loop works “often” but isn’t guaranteed correct by the Bubble Sort specification itself.
