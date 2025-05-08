#include <cuda_runtime.h>
#include <iostream>

__global__ void matmul(int* A, int* B, int* C, int N) {
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if (Row < N && Col < N) {
        int Pvalue = 0;
        for (int k = 0; k < N; k++) {
            Pvalue += A[Row * N + k] * B[k * N + Col];
        }
        C[Row * N + Col] = Pvalue;
    }
}

int main() {
    int N = 512;
    int size = N * N * sizeof(int);

    int* A;
    int* B;
    int* C;
    int* dev_A;
    int* dev_B;
    int* dev_C;

    // Allocate host and device memory
    cudaMallocHost(&A, size);
    cudaMallocHost(&B, size);
    cudaMallocHost(&C, size);
    cudaMalloc(&dev_A, size);
    cudaMalloc(&dev_B, size);
    cudaMalloc(&dev_C, size);

    // Initialize matrices A and B
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = i * N + j;
            B[i * N + j] = j * N + i;
        }
    }

    // Copy matrices A and B to device
    cudaMemcpy(dev_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, size, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 dimBlock(16, 16);
    dim3 dimGrid(N / dimBlock.x, N / dimBlock.y);
    matmul<<<dimGrid, dimBlock>>>(dev_A, dev_B, dev_C, N);

    // Copy result back to host
    cudaMemcpy(C, dev_C, size, cudaMemcpyDeviceToHost);

    // Print first 10x10 section of the result matrix
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            std::cout << C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    // Free memory
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);

    return 0;
}

















































**Explaining CUDA Matrix Multiplication**

The user wants to know how to run a CUDA matrix multiplication program and get an overview of the CUDA system. First, compile and run with the `nvcc matmul.cu -o matmul` command and `./matmul`. Note that OpenMP isn't relevant here because this is a CUDA program.

Then, I'll explain how the kernel works, how host code interacts with GPU memory, and how thread indexing works. I’ll use N=4 for matrix size with block dimensions 2x2 and grid layout 2x2 for a practical example.


Thought for 5 seconds


Here’s everything you need to compile & run your CUDA matrix‑multiply, a note about OpenMP (which isn’t used here), and a small worked example to see exactly what each GPU thread is doing.

---

## 1. Compile & Run Commands

Assuming you’ve saved your code as `matmul.cu`:

```bash
# 1. Compile with nvcc (the NVIDIA CUDA compiler)
nvcc matmul.cu -o matmul

# 2. Run the executable
./matmul
```

You should then see the upper‑left 10×10 block of the N×N result printed to the console.

---

## 2. OpenMP Note

This program uses **CUDA**, not OpenMP.

* **OpenMP** is a CPU‑side threading API in C/C++ (you’d use `#pragma omp parallel for` etc.).
* **CUDA** launches kernels on the GPU with the `<<<…>>>` syntax and uses `threadIdx`, `blockIdx`, etc.
  Any `#pragma omp` directives here would be ignored (or cause a compile error), because the NVIDIA compiler only recognizes CUDA keywords.

---

## 3. How the CUDA Code Works

### 3.1. Kernel: `__global__ void matmul(...)`

```cpp
int Row = blockIdx.y * blockDim.y + threadIdx.y;
int Col = blockIdx.x * blockDim.x + threadIdx.x;

if (Row < N && Col < N) {
  int Pvalue = 0;
  for (int k = 0; k < N; k++) {
    Pvalue += A[Row * N + k] * B[k * N + Col];
  }
  C[Row * N + Col] = Pvalue;
}
```

* **`blockDim`**: dimensions (x,y) of each block (here 16×16).
* **`blockIdx`**: which block in the grid this thread is in.
* **`threadIdx`**: thread’s coordinates (x,y) within its block.
* So

  $$
     \text{Row} = \bigl(\text{blockIdx.y}\times\text{blockDim.y}\bigr) + \text{threadIdx.y},
     \quad
     \text{Col} = \bigl(\text{blockIdx.x}\times\text{blockDim.x}\bigr) + \text{threadIdx.x}.
  $$
* Each thread computes one output element $C[\text{Row},\text{Col}]$ by “dot‑product” summing over $k=0\ldots N-1$.

### 3.2. Host Setup & Launch

1. **Pinned host memory**

   ```cpp
   cudaMallocHost(&A, size);  // same for B, C
   ```

   Page‑locked host buffers give faster transfers to/from the GPU.

2. **Device allocation**

   ```cpp
   cudaMalloc(&dev_A, size);
   cudaMalloc(&dev_B, size);
   cudaMalloc(&dev_C, size);
   ```

3. **Initialize**

   ```cpp
   A[i*N + j] = i*N + j;
   B[i*N + j] = j*N + i;
   ```

   (just an easy test pattern).

4. **Copy to device**

   ```cpp
   cudaMemcpy(dev_A, A, size, cudaMemcpyHostToDevice);
   cudaMemcpy(dev_B, B, size, cudaMemcpyHostToDevice);
   ```

5. **Launch**

   ```cpp
   dim3 dimBlock(16, 16);
   dim3 dimGrid(N/16, N/16);
   matmul<<<dimGrid, dimBlock>>>(dev_A, dev_B, dev_C, N);
   ```

   * Here `N=512`, so you get a 32×32 block grid, each block with 256 threads.

6. **Copy back**

   ```cpp
   cudaMemcpy(C, dev_C, size, cudaMemcpyDeviceToHost);
   ```

7. **Print & Cleanup**
   You print the top‑left 10×10 corner, then free all CPU and GPU buffers.

---

## 4. Worked Example (N = 3)

Let’s shrink to $N=3$, block size 2×2 (so grid is 2×2) and walk through the first few threads.

### 4.1. Input Matrices

```
A = [ [ 0, 1, 2 ],
      [ 3, 4, 5 ],
      [ 6, 7, 8 ] ]

B = [ [ 0, 3, 6 ],
      [ 1, 4, 7 ],
      [ 2, 5, 8 ] ]
```

### 4.2. Grid & Thread Indexing

* **dimBlock = (2,2)** → each block has threads with `(threadIdx.x, threadIdx.y)` in `{0,1}×{0,1}`.
* **dimGrid = (ceil(3/2)=2, ceil(3/2)=2)** → blocks with `blockIdx.x,blockIdx.y` in `{0,1}×{0,1}`.

| blockIdx.y | blockIdx.x | covered Rows | covered Cols |
| :--------: | :--------: | :----------: | :----------: |
|      0     |      0     |      0,1     |      0,1     |
|      0     |      1     |      0,1     |       2      |
|      1     |      0     |       2      |      0,1     |
|      1     |      1     |       2      |       2      |

### 4.3. Thread $(0,0)$ in Block $(0,0)$

* **Global Row** = $0*2 + 0 = 0$
* **Global Col** = $0*2 + 0 = 0$

Compute

$$
C[0,0] = \sum_{k=0}^2 A[0,k]\,B[k,0]
       = 0*0 + 1*1 + 2*2 = 0 + 1 + 4 = 5.
$$

### 4.4. Thread $(1,0)$ in Block $(0,0)$

* **Global Row** = $0*2 + 0 = 0$
* **Global Col** = $0*2 + 1 = 1$

Compute

$$
C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
       = 0*3 + 1*4 + 2*5 = 0 + 4 + 10 = 14.
$$

### 4.5. Thread $(0,1)$ in Block $(0,0)$

* **Row** = $1*2 + 0 = 2$? No—watch carefully:

  * `blockIdx.y=0`, `threadIdx.y=1` → Row = $0*2+1=1$
  * `blockIdx.x=0`, `threadIdx.x=0` → Col = $0*2+0=0$

Compute

$$
C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
       = 3*0 + 4*1 + 5*2 = 0 + 4 + 10 = 14.
$$

…and so on for every thread. After all blocks finish, you’ll have computed the full 3×3 product:

```
C = [ [ 5, 14, 23 ],
      [14, 50, 86 ],
      [23, 86,149 ] ]
```

Your code prints the upper 10×10 of a 512×512 result; for $N=3$ you’d see:

```
5 14 23
14 50 86
23 86 149
```

---

**Key takeaways:**

* **CUDA** uses `<<<grid,block>>>` to dispatch thousands of tiny GPU threads.
* Each thread’s `Row,Col` is computed from `blockIdx` and `threadIdx`.
* There's **no** OpenMP here—parallelism lives entirely on the GPU.
* You get massive speedups on large $N$ by overlapping thousands of multiply‑adds in hardware.
