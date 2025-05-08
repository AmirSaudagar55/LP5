#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

int parallelMin(vector<int> vec) {
    int min_val = vec[0];
    #pragma omp parallel for
    for (int i = 1; i < vec.size(); i++) {
        if (vec[i] < min_val) {
            min_val = vec[i];
        }
    }
    return min_val;
}

int parallelMax(vector<int> vec) {
    int max_val = vec[0];
    #pragma omp parallel for
    for (int i = 1; i < vec.size(); i++) {
        if (vec[i] > max_val) {
            max_val = vec[i];
        }
    }
    return max_val;
}

int parallelSum(vector<int> vec) {
    int sum = 0;
    #pragma omp parallel for
    for (int i = 0; i < vec.size(); i++) {
        sum += vec[i];
    }
    return sum;
}

float parallelAverage(vector<int> vec) {
    int sum = parallelSum(vec);
    float avg = float(sum) / vec.size();
    return avg;
}

int main() {
    int n;
    cout << "Enter the number of elements: ";
    cin >> n;

    vector<int> vec(n);
    cout << "Enter the elements: ";
    for (int i = 0; i < n; ++i) {
        cin >> vec[i];
    }

    int min_val = parallelMin(vec);
    cout << "Minimum value: " << min_val << endl;

    int max_val = parallelMax(vec);
    cout << "Maximum value: " << max_val << endl;

    int sum = parallelSum(vec);
    cout << "Sum of values: " << sum << endl;

    float avg = parallelAverage(vec);
    cout << "Average of values: " << avg << endl;

    return 0;
}




















































**Explaining OpenMP syntax**

The user wants an explanation of OpenMP syntax with C++, including pragmas like `#pragma omp parallel for`. I'll need to cover the benefits of parallel processing, how it helps with performance, and the potential risks such as data races. I should also highlight the importance of using the reduction clause to avoid these issues. For example, in the vector `[3, 1, 4, 2]`, parallel iteration can cause race conditions, unless reduction is used to ensure correct results. I’ll explain and walk through an example step by step, using four threads.


Thought for 4 seconds


Below is a detailed walkthrough of how OpenMP is used in your min/max/sum/average routines, why each pragma is there (and what pitfalls to watch for), followed by a fully worked example on a small array.

---

## 1. The OpenMP Pragmas

```cpp
#pragma omp parallel for
for (int i = 1; i < vec.size(); i++) {
    if (vec[i] < min_val) {
        min_val = vec[i];
    }
}
```

* **`#pragma omp parallel for`**:

  * Tells the compiler to spawn a team of threads and divide the `for`‑loop’s iterations among them.
  * Each thread gets its own private loop index `i`, but by default shares all other variables (like `min_val`) with the team.
  * At the end of the loop there is an implicit barrier—no thread moves on until all threads finish their chunk.

### Why This Alone Is Unsafe

Because `min_val`, `max_val`, and `sum` are **shared** and multiple threads may write them simultaneously, you introduce **data races**. In practice this may:

* Overwrite one thread’s update with another.
* Yield incorrect or unpredictable results.

---

## 2. Proper Parallel Reductions

OpenMP provides a neat way to do these safe, race‑free computations: the **`reduction`** clause. For example:

```cpp
int parallelMin(const vector<int>& vec) {
    int min_val = vec[0];
    #pragma omp parallel for reduction(min: min_val)
    for (int i = 1; i < vec.size(); i++) {
        if (vec[i] < min_val) {
            min_val = vec[i];
        }
    }
    return min_val;
}
```

* **`reduction(min: min_val)`**

  * Each thread keeps a private copy of `min_val`.
  * At the end of the parallel loop, OpenMP takes the **minimum** across all thread‑locals and stores it back into the shared `min_val`.
  * Eliminates data races without explicit locks.

Similarly:

```cpp
#pragma omp parallel for reduction(max: max_val)
#pragma omp parallel for reduction(+: sum)
```

---

## 3. Practical Flow in Code

1. **Initialization**

   ```cpp
   int min_val = vec[0];
   ```

   Everyone agrees on the starting value (the first element).

2. **Parallel Loop**

   ```cpp
   #pragma omp parallel for reduction(min: min_val)
   for (int i = 1; i < vec.size(); ++i) {
       min_val = std::min(min_val, vec[i]);
   }
   ```

   * OpenMP splits `i=1…N-1` among threads.
   * Each thread updates its own **local** `min_val`.

3. **Reduction**
   Once all threads finish, OpenMP reduces all locals via `min` and writes the overall minimum back to the original `min_val`.

4. **Return**

   ```cpp
   return min_val;
   ```

---

## 4. Step‑by‑Step Example

### Input

```
n = 6  
vec = [ 8,  3, 10,  5,  2,  7 ]
```

We’ll run `parallelMin`, `parallelMax`, `parallelSum`, and `parallelAverage` with **4 threads**.

---

### 4.1 parallelMin

```cpp
int min_val = vec[0];  // min_val = 8

#pragma omp parallel for reduction(min: min_val)
for (i = 1..5) {
    min_val = std::min(min_val, vec[i]);
}
```

**Thread assignments** (one possible distribution):

* **Thread 0** handles `i=1,2` → sees `3`, `10`

  * local\_min starts at 8 → min(8,3)=3 → min(3,10)=3
* **Thread 1** handles `i=3` → sees `5`

  * local\_min starts at 8 → min(8,5)=5
* **Thread 2** handles `i=4` → sees `2`

  * local\_min starts at 8 → min(8,2)=2
* **Thread 3** handles `i=5` → sees `7`

  * local\_min starts at 8 → min(8,7)=7

**Reduction step:** take min of {3, 5, 2, 7, (initial 8)} = **2**
→ `parallelMin` returns **2**.

---

### 4.2 parallelMax

```cpp
int max_val = vec[0];  // max_val = 8

#pragma omp parallel for reduction(max: max_val)
for (i = 1..5) {
    max_val = std::max(max_val, vec[i]);
}
```

**Per‑thread locals**:

* T0 (i=1,2): max(8,3)=8 → max(8,10)=10
* T1 (i=3): max(8,5)=8
* T2 (i=4): max(8,2)=8
* T3 (i=5): max(8,7)=8

**Reduction**: max of {10,8,8,8,8} = **10**
→ `parallelMax` returns **10**.

---

### 4.3 parallelSum

```cpp
int sum = 0;

#pragma omp parallel for reduction(+: sum)
for (i = 0..5) {
    sum += vec[i];
}
```

**Partials**:

* T0 (i=0,1): 8 + 3 = 11
* T1 (i=2,3): 10 + 5 = 15
* T2 (i=4): 2
* T3 (i=5): 7

**Reduction** (addition): 11 + 15 + 2 + 7 = **35**
→ `parallelSum` returns **35**.

---

### 4.4 parallelAverage

```cpp
float avg = float(parallelSum(vec)) / n;  // 35 / 6 ≈ 5.8333
```

→ **5.8333**

---

## 5. Summary of Results

| Function          | Value  |
| ----------------- | ------ |
| `parallelMin`     | 2      |
| `parallelMax`     | 10     |
| `parallelSum`     | 35     |
| `parallelAverage` | 5.8333 |

---

### Why This Matters

* **Speed**: On very large vectors, these reductions scale nearly linearly with thread count.
* **Correctness**: Using `reduction(...)` guarantees a race‑free combine of partial results.
* **Simplicity**: Just one pragma and clause swap turns many lines of mutex code into a clean, high‑level directive.

Feel free to integrate these patterns into your code wherever you need fast, thread‑safe min/max/sum/average computations!
