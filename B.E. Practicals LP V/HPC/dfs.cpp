#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>

using namespace std;

struct Graph {
    int V;
    vector<vector<int>> adj;

    Graph(int V) {
        this->V = V;
        adj.resize(V);
    }

    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    void DFS(int start) {
        vector<bool> visited(V, false);
        #pragma omp parallel
        {
            #pragma omp single nowait
            {
                DFSUtil(start, visited);
            }
        }
        cout << endl;
    }

    void DFSUtil(int u, vector<bool>& visited) {
        visited[u] = true;
        cout << u << " ";

        # pragma omp parallel for
        for (int i = 0; i < adj[u].size(); i++) {
            int v = adj[u][i];
            if (!visited[v]) {
                DFSUtil(v, visited);
            }
        }
    }
};
int main() {
    int V;
    cout << "Enter the number of vertices: ";
    cin >> V;

    Graph g(V);

    int edgeCount;
    cout << "Enter the number of edges: ";
    cin >> edgeCount;

    cout << "Enter the edges (in format 'source destination'): \n";
    for (int i = 0; i < edgeCount; i++) {
        int u, v;
        cin >> u >> v;
        g.addEdge(u, v);
    }

    cout << "DFS traversal starting from node 0: ";
    g.DFS(0);


    return 0;
}








































































Thought for a couple of seconds


Below is a breakdown of the OpenMP directives in your parallel DFS code, followed by a tiny worked‑through example so you can see how threads collaborate in practice.

---

## 1. OpenMP Primer

OpenMP is a simple way to turn sections of your C/C++ code into multi‑threaded regions. You sprinkle in `#pragma omp ...` lines, and the compiler/runtime take care of spawning threads, dividing work, and synchronizing.

---

## 2. The Pragmas in Your DFS

```cpp
void DFS(int start) {
    vector<bool> visited(V, false);

    #pragma omp parallel
    {
        #pragma omp single nowait
        {
            DFSUtil(start, visited);
        }
    }
    cout << endl;
}
```

1. **`#pragma omp parallel`**

   * Creates a team of threads. All threads begin executing the code **inside** the parallel block.
   * By default, variables declared outside (like `visited`) are shared among threads; loop indices or variables declared inside are private to each thread.

2. **`#pragma omp single nowait`**

   * Within that team, exactly **one** thread (any one) executes the **single** block (the initial call to `DFSUtil`).
   * **`nowait`** means: once that one thread has entered the single block, the other threads do **not** wait at an implicit barrier—they immediately continue past the single directive (in this case, the parallel region ends for them).

   Effectively, thread 0 (say) will call `DFSUtil(start, visited);` while threads 1…N skip ahead—ready to join any further parallel work inside `DFSUtil`.

---

Inside your recursive helper:

```cpp
void DFSUtil(int u, vector<bool>& visited) {
    visited[u] = true;
    cout << u << " ";

    #pragma omp parallel for
    for (int i = 0; i < adj[u].size(); i++) {
        int v = adj[u][i];
        if (!visited[v]) {
            DFSUtil(v, visited);
        }
    }
}
```

3. **`#pragma omp parallel for`**

   * Again spawns a team of threads for the **for**‑loop that follows.
   * The loop iterations (`i = 0…adj[u].size()-1`) are divided among threads. Each thread picks a subset of neighbors to process in parallel.

   So, when exploring node `u`, each neighbor check `if (!visited[v]) DFSUtil(v, visited);` can happen concurrently across neighbors.

---

## 3. How It Plays Out: A Mini Example

Consider this tiny undirected graph:

```
   0
  / \
 1   2
 |   |
 3   4
```

Adjacency lists:

* `adj[0] = {1,2}`
* `adj[1] = {0,3}`
* `adj[2] = {0,4}`
* `adj[3] = {1}`
* `adj[4] = {2}`

### Serial DFS Traversal

Starting at `0`, a serial DFS might visit in order:
0 → 1 → 3 → 2 → 4

### Parallelized DFS

1. **Entering `DFS(0)`**

   * **All threads** spawn (`#pragma omp parallel`), but only one executes the `single` block to call `DFSUtil(0,…)`.

2. **In `DFSUtil(0,…)`**

   * `visited[0]=true`, print `0`.

   * Hit `#pragma omp parallel for` over `i=0,1` for neighbors `{1,2}`.

   * **Thread A** takes `i=0 → v=1`: sees `visited[1]=false`, calls `DFSUtil(1,…)`.

   * **Thread B** takes `i=1 → v=2`: sees `visited[2]=false`, calls `DFSUtil(2,…)`.

   These two calls happen **concurrently**:

   * **Thread A** in `DFSUtil(1)`:

     * Mark and print `1`.
     * Parallel‑for its neighbor list `{0,3}`.

       * One iteration sees `0` already visited → skip.
       * Other iteration `v=3` → `DFSUtil(3)` prints `3`.

   * **Thread B** in `DFSUtil(2)`:

     * Mark and print `2`.
     * Parallel‑for `{0,4}`:

       * Skip `0`, call `DFSUtil(4)` to print `4`.

3. **Possible Printed Order**
   Because calls overlap, you might see something like:

   ```
   0 1 2 3 4 
   ```

   or even

   ```
   0 2 1 4 3 
   ```

   The exact interleaving depends on thread scheduling.

---

## 4. Key Points & Caveats

* **Shared `visited`**: All threads write to the same `visited` array without locking. This risks two threads recursing into the same `v` concurrently. You’ll need a critical section or atomic check to prevent duplicate visits.
* **Recursive Parallel Regions**: Each `parallel for` spawns threads anew. This can be expensive if the adjacency lists are small.
* **Scalability**: Best when each node has many neighbors, so each parallel‑for does enough work to outweigh thread‑creation overhead.

---

### In Practice

* For **dense graphs** or heavy per‑neighbor processing, this pattern can speed up DFS by overlapping the exploration of different branches.
* Always guard shared updates (like marking `visited[v] = true`) with `#pragma omp critical` or use atomic operations to avoid race conditions.

This shows how a few well‑placed OpenMP pragmas let you recursively explore a graph in parallel—dividing up neighbor visits across threads while still performing a depth‑first search.
