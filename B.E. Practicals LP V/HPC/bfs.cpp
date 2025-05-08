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

    void BFS(int start) {
        vector<bool> visited(V, false);
        queue<int> q;
        visited[start] = true;
        q.push(start);

        while (!q.empty()) {
            int u = q.front();
            q.pop();
            cout << u << " ";

            #pragma omp parallel for
            for (int i = 0; i < adj[u].size(); i++) {
                int v = adj[u][i];
                if (!visited[v]) {
                    #pragma omp critical
                    {
                        visited[v] = true;
                        q.push(v);
                    }
                }
            }
        }
        cout << endl;
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

    cout << "BFS traversal starting from node 0: ";
    g.BFS(0);

    return 0;
}








































































Thought for a couple of seconds


Here’s what’s happening in your BFS code with OpenMP, step by step:

---

## 1. What is OpenMP?

OpenMP is a set of compiler directives, library routines, and environment variables that let you write shared‑memory parallel code in C/C++ (and Fortran) with minimal changes. You sprinkle “#pragma omp …” lines around loops or regions you want to run in parallel, and the OpenMP runtime takes care of creating threads, dividing work, and synchronizing them.

---

## 2. The Pragmas in Your BFS

```cpp
#pragma omp parallel for
for (int i = 0; i < adj[u].size(); i++) {
    int v = adj[u][i];
    if (!visited[v]) {
        #pragma omp critical
        {
            visited[v] = true;
            q.push(v);
        }
    }
}
```

1. **`#pragma omp parallel for`**

   * Tells the compiler: “Spawn a team of threads and split the iterations of the following `for` loop evenly among them.”
   * Each thread gets its own loop‑index `i` and executes its chunk of neighbors `adj[u][i]` concurrently.

2. **Why parallelize this loop?**

   * In a standard (serial) BFS, for each dequeued node `u`, you scan through all its neighbors one by one and enqueue any unvisited ones.
   * If `u` has many neighbors, you can check “visited” and enqueue them in parallel.

3. **Race Conditions & the Critical Section**

   * Multiple threads might simultaneously find the same neighbor `v` unvisited and try to:

     ```cpp
     visited[v] = true;
     q.push(v);
     ```
   * Without protection, two threads could both enqueue `v` (duplicate work) or corrupt the queue’s internal state.
   * **`#pragma omp critical`** creates a mutex around the following block so *only one thread at a time* can execute it, ensuring:

     * `visited[v]` is set exactly once.
     * `v` is pushed into the queue exactly once.

4. **Thread‑Local Copies**

   * Note: the queue `q` and the `visited` vector are shared by default under OpenMP.
   * The loop‑index `i` and the local variable `v` are private to each thread (OpenMP makes loop indices private automatically).

---

## 3. Practical Walk‑through with a Tiny Graph

Let’s say we have this undirected graph:

```
    (0)
    / \
  (1)--(2)
    \  
    (3)
```

Adjacency lists:

* `adj[0] = {1,2}`
* `adj[1] = {0,2,3}`
* …

### Serial BFS from 0

1. Dequeue `0`, mark visited\[0]=true
2. Scan neighbors `{1,2}` one by one:

   * see `1` unvisited → mark visited\[1]=true, enqueue `1`
   * see `2` unvisited → mark visited\[2]=true, enqueue `2`
3. Dequeue `1`, scan `{0,2,3}`, enqueue `3`
4. … and so on.

### Parallelized Neighbor Check

At node `u = 0`:

```cpp
#pragma omp parallel for
for i in {0,1}:
  v = adj[0][i]     // thread‑0 might take i=0 → v=1; thread‑1 i=1 → v=2
  if (!visited[v]) { 
    #pragma omp critical
      visited[v] = true; q.push(v);
  }
```

* **Thread A** checks `v=1` and enters critical section to mark & enqueue it.
* **Thread B**, at (nearly) the same time, checks `v=2` and enters the same critical section (one after the other) to mark & enqueue it.
* End result after this loop (still in parallel region) is the queue contains `1,2` (order may vary).

Then you exit the parallel region, go back to the serial `while(!q.empty())` loop, dequeue the next node, and repeat.

---

## 4. Pros & Cons

* **Pros:**

  * If a node has many neighbors (say hundreds), you can leverage multiple cores to check them faster, especially if the “visited” array check or neighbor‑processing work is heavy.

* **Cons:**

  * Overhead of spawning threads and entering/exiting critical sections may outweigh the gains for small adjacency lists.
  * The single critical region is a serialization point: if many threads hit it at once, they’ll queue up, so you don’t get full parallel speed on the enqueue step itself.

---

## 5. When It Helps Most

* **Large, dense graphs** where many nodes have dozens or hundreds of neighbors.
* **Heavy neighbor‑processing** (e.g. computing some property of each neighbor before enqueue) so each thread does enough work to amortize the cost of threading and locking.

---

### Summary

* **`#pragma omp parallel for`** splits your neighbor‑scanning loop across threads.
* **`#pragma omp critical`** guards the enqueue & visited‑flag update to avoid races.
* Together, they let you do the bulk of your neighbor checks in parallel while still safely updating shared state.
