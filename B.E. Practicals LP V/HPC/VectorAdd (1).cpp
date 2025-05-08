#include <iostream>
#include <cuda_runtime.h>

// Correct kernel declaration
__global__ void addVectors(int* A, int* B, int* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int n = 1000000;
    int size = n * sizeof(int);

    int* A, * B, * C;
    int* dev_A, * dev_B, * dev_C;

    // Allocate pinned host memory
    cudaMallocHost(&A, size);
    cudaMallocHost(&B, size);
    cudaMallocHost(&C, size);

    // Initialize vectors A and B
    for (int i = 0; i < n; i++) {
        A[i] = i;
        B[i] = i * 2;
    }

    // Allocate device memory
    cudaMalloc(&dev_A, size);
    cudaMalloc(&dev_B, size);
    cudaMalloc(&dev_C, size);

    // Copy host data to device
    cudaMemcpy(dev_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, size, cudaMemcpyHostToDevice);

    // Launch the kernel
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    addVectors<<<numBlocks, blockSize>>>(dev_A, dev_B, dev_C, n);

    // Copy result back to host
    cudaMemcpy(C, dev_C, size, cudaMemcpyDeviceToHost);

    // Print first 10 elements of the result
    for (int i = 0; i < 10; i++) {
        std::cout << C[i] << " ";
    }
    std::cout << std::endl;

    // Free memory
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);

    return 0;
}



















































**Clarifying CUDA commands**

It seems like the user has a bit of confusion, mixing OpenMP and CUDA. The program is written in CUDA C++, so no OpenMP syntax is involved here. They probably meant to ask about CUDA-specific commands. To run the program, you'd use `nvcc` to compile, and then execute with `./exec`.

I think they might be interested in understanding parallelism, but OpenMP isn’t relevant here. I’ll clarify that CUDA uses its own model for parallel execution, not OpenMP.


Thought for 5 seconds


Below is everything you need to compile and run your CUDA vector‐add program, followed by a note on OpenMP (which isn’t used here), and then a step‐by‐step walk‑through of how the CUDA code works on a small example.

---

## 1. Compile & Run Commands

Assuming your file is named `vectorAdd.cu`:

```bash
# 1. Compile with nvcc (the NVIDIA CUDA compiler)
nvcc vectorAdd.cu -o vectorAdd

# 2. Run the executable
./vectorAdd
```

You should see the first 10 results printed, e.g.:

```
0 3 6 9 12 15 18 21 24 27
```

---

## 2. “OpenMP Syntax” Note

There is **no OpenMP** in this program—OpenMP is a CPU‑side threading API in C/C++. Here you’re using **CUDA**, which offloads parallel work to the **GPU** via `__global__` kernels and `nvcc`. So any `#pragma omp` directives would be ignored or invalid. GPU parallelism is managed by thread‑ and block‑indexes inside your `addVectors<<<…>>>` launch.

---

## 3. How the CUDA Code Works (Practical Walk‑through)

Let’s run through a tiny example with `n = 8` so you can see each step:

### A. Host Arrays Initialization

```cpp
int n = 8;
int size = n * sizeof(int);
// A[i] = i,    B[i] = 2*i
// A = [0,1,2,3,4,5,6,7]
// B = [0,2,4,6,8,10,12,14]
```

### B. Allocate & Initialize Pinned Host Memory

```cpp
cudaMallocHost(&A, size);
cudaMallocHost(&B, size);
cudaMallocHost(&C, size);
```

‑ Pinned (“page‑locked”) memory gives faster DMA transfers to/from the GPU.

### C. Allocate Device Memory

```cpp
cudaMalloc(&dev_A, size);
cudaMalloc(&dev_B, size);
cudaMalloc(&dev_C, size);
```

### D. Copy Host → Device

```cpp
cudaMemcpy(dev_A, A, size, cudaMemcpyHostToDevice);
cudaMemcpy(dev_B, B, size, cudaMemcpyHostToDevice);
```

Now the GPU has its own copies of A and B.

### E. Launch the Kernel

```cpp
int blockSize = 4;                 // choose 4 threads per block
int numBlocks = (n + blockSize-1)/blockSize;  
// for n=8: numBlocks = (8+3)/4 = 2
addVectors<<< numBlocks, blockSize >>>(dev_A, dev_B, dev_C, n);
```

* **Thread indexing**
  Each CUDA thread computes one output index `i`:

  ```cpp
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
     C[i] = A[i] + B[i];
  ```

  * Block 0, threads 0–3 handle i = 0,1,2,3
  * Block 1, threads 0–3 handle i = 4,5,6,7

### F. Copy Device → Host

```cpp
cudaMemcpy(C, dev_C, size, cudaMemcpyDeviceToHost);
```

Now your host array `C` contains the computed sums.

### G. Inspect the Results

For our tiny example:

|  i  | A\[i] | B\[i] | C\[i] = A\[i]+B\[i] |
| :-: | :---: | :---: | :-----------------: |
|  0  |   0   |   0   |          0          |
|  1  |   1   |   2   |          3          |
|  2  |   2   |   4   |          6          |
|  3  |   3   |   6   |          9          |
|  4  |   4   |   8   |          12         |
|  5  |   5   |   10  |          15         |
|  6  |   6   |   12  |          18         |
|  7  |   7   |   14  |          21         |

Printing the first 8 (or 10) elements yields:

```
0 3 6 9 12 15 18 21 ...
```

### H. Cleanup

Always free both device and pinned host memory:

```cpp
cudaFree(dev_A);
cudaFree(dev_B);
cudaFree(dev_C);
cudaFreeHost(A);
cudaFreeHost(B);
cudaFreeHost(C);
```

---

**In summary:**

* **Compile** with `nvcc`, **run** the produced binary.
* This is **CUDA** code, not OpenMP.
* The kernel `addVectors` launches thousands of lightweight GPU threads, each adding one pair of elements in parallel.
* Pinned host memory (`cudaMallocHost`) + explicit `cudaMemcpy` calls orchestrate data movement between CPU and GPU.
* You get massive speed‑ups over a serial CPU loop once you scale to millions of elements.
