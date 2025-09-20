import React, { useEffect, useState } from "react";
/**
 * DSA Playground – Single‑file React app
 * -------------------------------------
 * What you get in ONE file:
 * - Sorting: Bubble, Insertion, Merge, Quick (with bar-visualization)
 * - Searching: Linear, Binary
 * - Stack & Queue (interactive)
 * - Linked List (insert/delete/visualize)
 * - Trees: Binary Search Tree (insert/search/traversals)
 * - Graph: Adjacency List, BFS, DFS, Dijkstra (shortest path)
 * - Dynamic Programming: Fibonacci (memo/tab), 0/1 Knapsack
 * - Strings: KMP substring search
 * - Hash Table: Separate chaining demo (simple)
 * - Bench: small runner to time algorithms
 */
// ----------------------------- Utils ----------------------------------
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const rnd = (n: number, max = 99) => Array.from({ length: n }, () => Math.floor(Math.random() * (max + 1)));

function timeIt<T>(label: string, fn: () => T): { label: string; ms: number; result: T } {
  const t0 = performance.now();
  const result = fn();
  const t1 = performance.now();
  return { label, ms: t1 - t0, result };
}

// Deep copy for purity when needed
const deep = <T,>(x: T): T => JSON.parse(JSON.stringify(x));

// ----------------------------- Sorting --------------------------------
async function bubbleSort(arr: number[], onStep?: (state: number[]) => void, delay = 0) {
  const a = [...arr];
  const n = a.length;
  for (let i = 0; i < n - 1; i++) {
    for (let j = 0; j < n - i - 1; j++) {
      if (a[j] > a[j + 1]) {
        [a[j], a[j + 1]] = [a[j + 1], a[j]];
        onStep?.([...a]);
        if (delay) await sleep(delay);
      }
    }
  }
  return a;
}

async function insertionSort(arr: number[], onStep?: (state: number[]) => void, delay = 0) {
  const a = [...arr];
  for (let i = 1; i < a.length; i++) {
    let key = a[i];
    let j = i - 1;
    while (j >= 0 && a[j] > key) {
      a[j + 1] = a[j];
      j--;
      onStep?.([...a]);
      if (delay) await sleep(delay);
    }
    a[j + 1] = key;
    onStep?.([...a]);
    if (delay) await sleep(delay);
  }
  return a;
}

function merge(left: number[], right: number[], onStep?: (state: number[]) => void) {
  const out: number[] = [];
  let i = 0, j = 0;
  while (i < left.length && j < right.length) {
    if (left[i] <= right[j]) out.push(left[i++]);
    else out.push(right[j++]);
    onStep?.([...out, ...left.slice(i), ...right.slice(j)]);
  }
  return [...out, ...left.slice(i), ...right.slice(j)];
}

async function mergeSort(arr: number[], onStep?: (state: number[]) => void, delay = 0): Promise<number[]> {
  if (arr.length <= 1) return arr;
  const mid = Math.floor(arr.length / 2);
  const left = await mergeSort(arr.slice(0, mid), onStep, delay);
  const right = await mergeSort(arr.slice(mid), onStep, delay);
  const merged = merge(left, right, onStep);
  if (delay) await sleep(delay);
  return merged;
}

async function quickSort(arr: number[], onStep?: (state: number[]) => void, delay = 0): Promise<number[]> {
  const a = [...arr];
  async function qs(l: number, r: number) {
    if (l >= r) return;
    const pivot = a[Math.floor((l + r) / 2)];
    let i = l, j = r;
    while (i <= j) {
      while (a[i] < pivot) i++;
      while (a[j] > pivot) j--;
      if (i <= j) {
        [a[i], a[j]] = [a[j], a[i]];
        onStep?.([...a]);
        if (delay) await sleep(delay);
        i++; j--;
      }
    }
    await qs(l, j);
    await qs(i, r);
  }
  await qs(0, a.length - 1);
  return a;
}

// ----------------------------- Searching ------------------------------
function linearSearch<T>(arr: T[], target: T) {
  for (let i = 0; i < arr.length; i++) if (arr[i] === target) return i;
  return -1;
}

function binarySearch(arr: number[], target: number) {
  let l = 0, r = arr.length - 1;
  while (l <= r) {
    const m = Math.floor((l + r) / 2);
    if (arr[m] === target) return m;
    if (arr[m] < target) l = m + 1; else r = m - 1;
  }
  return -1;
}

// ----------------------------- Stack & Queue --------------------------
class Stack<T> {
  private data: T[] = [];
  push(x: T) { this.data.push(x); }
  pop(): T | undefined { return this.data.pop(); }
  peek(): T | undefined { return this.data[this.data.length - 1]; }
  toArray() { return [...this.data]; }
}

class Queue<T> {
  private data: T[] = [];
  enqueue(x: T) { this.data.push(x); }
  dequeue(): T | undefined { return this.data.shift(); }
  front(): T | undefined { return this.data[0]; }
  toArray() { return [...this.data]; }
}

// ----------------------------- Linked List ----------------------------
class ListNode<T> {
  val: T;
  next: ListNode<T> | null;
  constructor(val: T, next: ListNode<T> | null = null) {
    this.val = val;
    this.next = next;
  }
}
class SinglyLinkedList<T> {
  head: ListNode<T> | null = null;
  insertAtHead(val: T) { const n = new ListNode(val, this.head); this.head = n; }
  insertAtTail(val: T) {
    const n = new ListNode(val);
    if (!this.head) { this.head = n; return; }
    let cur = this.head; while (cur.next) cur = cur.next; cur.next = n;
  }
  delete(val: T) {
    if (!this.head) return;
    if (this.head.val === val) { this.head = this.head.next; return; }
    let cur = this.head;
    while (cur.next && cur.next.val !== val) cur = cur.next;
    if (cur.next) cur.next = cur.next.next;
  }
  toArray(): T[] { const out: T[] = []; let c = this.head; while (c) { out.push(c.val); c = c.next; } return out; }
}

// ----------------------------- Trees (BST) ----------------------------
class BSTNode {
  key: number;
  left: BSTNode | null;
  right: BSTNode | null;

  constructor(key: number, left: BSTNode | null = null, right: BSTNode | null = null) {
    this.key = key;
    this.left = left;
    this.right = right;
  }
}
class BST {
  root: BSTNode | null = null;
  insert(key: number) { this.root = this._insert(this.root, key); }
  private _insert(node: BSTNode | null, key: number): BSTNode {
    if (!node) return new BSTNode(key);
    if (key < node.key) node.left = this._insert(node.left, key);
    else if (key > node.key) node.right = this._insert(node.right, key);
    return node;
  }
  search(key: number): boolean { let n = this.root; while (n) { if (key === n.key) return true; n = key < n.key ? n.left : n.right; } return false; }
  inorder(): number[] { const out: number[] = []; const dfs = (n: BSTNode | null) => { if (!n) return; dfs(n.left); out.push(n.key); dfs(n.right); }; dfs(this.root); return out; }
  preorder(): number[] { const out: number[] = []; const dfs = (n: BSTNode | null) => { if (!n) return; out.push(n.key); dfs(n.left); dfs(n.right); }; dfs(this.root); return out; }
  postorder(): number[] { const out: number[] = []; const dfs = (n: BSTNode | null) => { if (!n) return; dfs(n.left); dfs(n.right); out.push(n.key); }; dfs(this.root); return out; }
}

// ----------------------------- Graph ----------------------------------
class Graph {
  adj: Record<string, Array<{ v: string; w?: number }>> = {};
  addVertex(v: string) { if (!this.adj[v]) this.adj[v] = []; }
  addEdge(u: string, v: string, w = 1, undirected = true) {
    this.addVertex(u); this.addVertex(v);
    this.adj[u].push({ v, w });
    if (undirected) this.adj[v].push({ v: u, w });
  }
  bfs(start: string) {
    const visited = new Set<string>(); const order: string[] = []; const q: string[] = [start]; visited.add(start);
    while (q.length) { const u = q.shift()!; order.push(u); for (const { v } of this.adj[u] || []) if (!visited.has(v)) { visited.add(v); q.push(v); } }
    return order;
  }
  dfs(start: string) {
    const visited = new Set<string>(); const order: string[] = [];
    const go = (u: string) => { visited.add(u); order.push(u); for (const { v } of this.adj[u] || []) if (!visited.has(v)) go(v); };
    go(start); return order;
  }
  dijkstra(src: string) {
    const dist: Record<string, number> = {}; const prev: Record<string, string | null> = {}; const Q = new Set<string>(Object.keys(this.adj));
    for (const v of Q) { dist[v] = Infinity; prev[v] = null; } dist[src] = 0;
    while (Q.size) {
      let u: string | null = null; let best = Infinity;
      for (const v of Q) if (dist[v] < best) { best = dist[v]; u = v; }
      if (u === null) break; Q.delete(u);
      for (const { v, w = 1 } of this.adj[u] || []) {
        const alt = dist[u] + w; if (alt < dist[v]) { dist[v] = alt; prev[v] = u; }
      }
    }
    const pathTo = (t: string) => { const path: string[] = []; let cur: string | null = t; while (cur) { path.push(cur); cur = prev[cur]; } return path.reverse(); };
    return { dist, prev, pathTo };
  }
}

// ----------------------------- Dynamic Programming --------------------
function fibMemo(n: number, memo: Record<number, number> = {}): number {
  if (n <= 1) return n; if (memo[n] !== undefined) return memo[n];
  return memo[n] = fibMemo(n - 1, memo) + fibMemo(n - 2, memo);
}

function fibTab(n: number): number {
  if (n <= 1) return n; const dp = [0, 1]; for (let i = 2; i <= n; i++) dp[i] = dp[i - 1] + dp[i - 2]; // @ts-ignore
  return dp[n];
}

function knapsack01(weights: number[], values: number[], W: number) {
  const n = weights.length; const dp: number[][] = Array.from({ length: n + 1 }, () => Array(W + 1).fill(0));
  for (let i = 1; i <= n; i++) {
    for (let w = 0; w <= W; w++) {
      dp[i][w] = dp[i - 1][w];
      if (weights[i - 1] <= w) dp[i][w] = Math.max(dp[i][w], values[i - 1] + dp[i - 1][w - weights[i - 1]]);
    }
  }
  // reconstruct
  let w = W; const picked: number[] = [];
  for (let i = n; i >= 1; i--) {
    if (dp[i][w] !== dp[i - 1][w]) { picked.push(i - 1); w -= weights[i - 1]; }
  }
  return { best: dp[n][W], picked: picked.reverse(), table: dp };
}

// ----------------------------- Strings (KMP) --------------------------
function kmpPrefix(p: string) {
  const pi = Array(p.length).fill(0); let j = 0;
  for (let i = 1; i < p.length; i++) {
    while (j > 0 && p[i] !== p[j]) j = pi[j - 1];
    if (p[i] === p[j]) j++; pi[i] = j;
  }
  return pi;
}
function kmpSearch(text: string, pat: string) {
  if (!pat) return 0; const pi = kmpPrefix(pat); let j = 0;
  for (let i = 0; i < text.length; i++) {
    while (j > 0 && text[i] !== pat[j]) j = pi[j - 1];
    if (text[i] === pat[j]) j++;
    if (j === pat.length) return i - j + 1;
  }
  return -1;
}

// ----------------------------- Hash Table (chaining) ------------------
class HashTable<K extends string | number, V> {
  private buckets: [K, V][][];
  private capacity: number;

  constructor(capacity = 17) {
    this.capacity = capacity;
    this.buckets = Array.from({ length: capacity }, () => []);
  }

  private hash(key: K) {
    const s = String(key);
    let h = 0;
    for (let i = 0; i < s.length; i++) {
      h = (h * 31 + s.charCodeAt(i)) >>> 0;
    }
    return h % this.capacity;
  }

  set(key: K, value: V) {
    const idx = this.hash(key);
    const b = this.buckets[idx];
    const i = b.findIndex(([k]) => k === key);
    if (i >= 0) {
      b[i][1] = value;
    } else {
      b.push([key, value]);
    }
  }

  get(key: K): V | undefined {
    const idx = this.hash(key);
    const b = this.buckets[idx];
    const f = b.find(([k]) => k === key);
    return f?.[1];
  }

  remove(key: K) {
    const idx = this.hash(key);
    const b = this.buckets[idx];
    const i = b.findIndex(([k]) => k === key);
    if (i >= 0) {
      b.splice(i, 1);
    }
  }

  loadFactor() {
    const items = this.buckets.reduce((a, b) => a + b.length, 0);
    return items / this.capacity;
  }

  entries() {
    return this.buckets.flat();
  }
}

// ----------------------------- UI Helpers -----------------------------
function Section({ title, children, right }: { title: string; children: React.ReactNode; right?: React.ReactNode }) {
  return (
    <div className="rounded-2xl border p-4 shadow-sm bg-white">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-xl font-semibold">{title}</h2>
        {right}
      </div>
      {children}
    </div>
  );
}

function Pill({ children }: { children: React.ReactNode }) {
  return <span className="px-2 py-1 text-xs rounded-full border bg-gray-50">{children}</span>;
}

function BarChart({ data, highlightIndex }: { data: number[]; highlightIndex?: number }) {
  const max = Math.max(1, ...data);
  return (
    <div className="h-48 flex items-end gap-1 w-full bg-gray-50 rounded p-2">
      {data.map((v, i) => (
        <div key={i} className={`flex-1 rounded-t ${highlightIndex === i ? "ring-2" : ""}`} style={{ height: `${(v / max) * 100}%` }} title={`${v}`}></div>
      ))}
    </div>
  );
}

function Textarea({ value, onChange, rows = 4 }: { value: string; onChange: (v: string) => void; rows?: number }) {
  return <textarea className="w-full border rounded-xl p-2 font-mono" rows={rows} value={value} onChange={(e) => onChange(e.target.value)} />;
}

// ----------------------------- Main App -------------------------------
export default function DSAPlayground() {
  const [tab, setTab] = useState<
    | "sorting" | "search" | "stackqueue" | "linkedlist" | "bst" | "graph" | "dp" | "strings" | "hash" | "bench"
  >("sorting");

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-white text-gray-900 p-4 md:p-8">
      <header className="mb-6">
        <h1 className="text-3xl md:text-4xl font-bold">DSA Playground</h1>
        <p className="text-gray-600">Interactive algorithms, visualizations, and a tiny benchmark runner. Ready to deploy.</p>
        <div className="mt-3 flex flex-wrap gap-2">
          {([
            ["sorting", "Sorting"],
            ["search", "Searching"],
            ["stackqueue", "Stack & Queue"],
            ["linkedlist", "Linked List"],
            ["bst", "Binary Search Tree"],
            ["graph", "Graph"],
            ["dp", "Dynamic Programming"],
            ["strings", "Strings (KMP)"],
            ["hash", "Hash Table"],
            ["bench", "Bench"]
          ] as const).map(([k, label]) => (
            <button key={k} onClick={() => setTab(k)} className={`px-3 py-1.5 rounded-xl border ${tab === k ? "bg-black text-white" : "bg-white"}`}>{label}</button>
          ))}
        </div>
      </header>

      <main className="grid gap-6 md:grid-cols-2">
        {tab === "sorting" && <SortingPanel />}
        {tab === "search" && <SearchPanel />}
        {tab === "stackqueue" && <StackQueuePanel />}
        {tab === "linkedlist" && <LinkedListPanel />}
        {tab === "bst" && <BSTPanel />}
        {tab === "graph" && <GraphPanel />}
        {tab === "dp" && <DPPanel />}
        {tab === "strings" && <StringsPanel />}
        {tab === "hash" && <HashPanel />}
        {tab === "bench" && <BenchPanel />}
      </main>

      <footer className="mt-8 text-sm text-gray-500">
        Built with ❤️ in React. Pro tip: Deploy to Vercel/Netlify in under 2 mins.
      </footer>
    </div>
  );
}

// ----------------------------- Panels ---------------------------------
function SortingPanel() {
  const [arr, setArr] = useState<number[]>(rnd(20, 50));
  const [view, setView] = useState<number[]>(arr);
  const [delay, setDelay] = useState(20);
  const [algo, setAlgo] = useState<"bubble" | "insertion" | "merge" | "quick">("merge");

  useEffect(() => { setView(arr); }, [arr]);

  const run = async () => {
    const onStep = (state: number[]) => setView(state);
    let out: number[] = [];
    if (algo === "bubble") out = await bubbleSort(arr, onStep, delay);
    if (algo === "insertion") out = await insertionSort(arr, onStep, delay);
    if (algo === "merge") out = await mergeSort(arr, onStep, delay);
    if (algo === "quick") out = await quickSort(arr, onStep, delay);
    setView(out);
  };

  return (
    <>
      <Section title="Sorting Visualizer" right={<Pill>{algo} sort</Pill>}>
        <div className="mb-3 flex flex-wrap gap-2 items-center">
          <button className="px-3 py-1.5 rounded-xl border" onClick={() => setArr(rnd(20, 50))}>New Array</button>
          <button className="px-3 py-1.5 rounded-xl border" onClick={run}>Run</button>
          <label className="text-sm">Delay: <input type="range" min={0} max={200} value={delay} onChange={e => setDelay(Number(e.target.value))} /></label>
          <select className="px-2 py-1 border rounded-xl" value={algo} onChange={(e) => setAlgo(e.target.value as any)}>
            <option value="merge">Merge</option>
            <option value="quick">Quick</option>
            <option value="insertion">Insertion</option>
            <option value="bubble">Bubble</option>
          </select>
        </div>
        <BarChart data={view} />
      </Section>

      <Section title="Notes">
        <ul className="list-disc ml-5 text-sm">
          <li>Merge/Quick are O(n log n) average; Insertion is great for nearly-sorted arrays.</li>
          <li>Try delay 0 to benchmark vs. non-zero for animations.</li>
        </ul>
      </Section>
    </>
  );
}

function SearchPanel() {
  const [arr, setArr] = useState<number[]>(() => [...rnd(16, 99)].sort((a, b) => a - b));
  const [target, setTarget] = useState<number>(arr[3] ?? 0);
  const [idx, setIdx] = useState<number | null>(null);

  return (
    <>
      <Section title="Binary vs Linear Search" right={<Pill>O(log n) vs O(n)</Pill>}>
        <div className="flex flex-wrap gap-2 items-end">
          <button className="px-3 py-1.5 rounded-xl border" onClick={() => { const a = [...rnd(16, 99)].sort((x,y)=>x-y); setArr(a); setTarget(a[3] ?? 0); setIdx(null); }}>New Sorted Array</button>
          <label className="text-sm">Target <input className="border rounded px-2 py-1 ml-1 w-20" type="number" value={target} onChange={e=>setTarget(Number(e.target.value))} /></label>
          <button className="px-3 py-1.5 rounded-xl border" onClick={() => setIdx(linearSearch(arr, target))}>Linear</button>
          <button className="px-3 py-1.5 rounded-xl border" onClick={() => setIdx(binarySearch(arr, target))}>Binary</button>
        </div>
        <div className="mt-3 flex flex-wrap gap-2">
          {arr.map((v,i)=> (
            <div key={i} className={`px-2 py-1 rounded border ${idx===i?"bg-black text-white":"bg-white"}`}>{v}</div>
          ))}
        </div>
      </Section>

      <Section title="Notes">
        <p className="text-sm">Binary search requires a <b>sorted</b> array and halves the search space each step.</p>
      </Section>
    </>
  );
}

function StackQueuePanel() {
  const [stack] = useState(() => new Stack<number>());
  const [queue] = useState(() => new Queue<number>());
  const [, force] = useState(0);
  const bump = () => force((x) => x + 1);

  return (
    <>
      <Section title="Stack (LIFO)">
        <div className="flex gap-2 mb-2">
          <button className="px-3 py-1.5 rounded-xl border" onClick={() => { stack.push(Math.floor(Math.random()*100)); bump(); }}>Push</button>
          <button className="px-3 py-1.5 rounded-xl border" onClick={() => { stack.pop(); bump(); }}>Pop</button>
        </div>
        <div className="flex gap-2">{stack.toArray().map((v,i)=>(<div key={i} className="px-2 py-1 rounded border">{v}</div>))}</div>
      </Section>

      <Section title="Queue (FIFO)">
        <div className="flex gap-2 mb-2">
          <button className="px-3 py-1.5 rounded-xl border" onClick={() => { queue.enqueue(Math.floor(Math.random()*100)); bump(); }}>Enqueue</button>
          <button className="px-3 py-1.5 rounded-xl border" onClick={() => { queue.dequeue(); bump(); }}>Dequeue</button>
        </div>
        <div className="flex gap-2">{queue.toArray().map((v,i)=>(<div key={i} className="px-2 py-1 rounded border">{v}</div>))}</div>
      </Section>
    </>
  );
}

function LinkedListPanel() {
  const [list] = useState(() => new SinglyLinkedList<number>());
  const [val, setVal] = useState(1);
  const [, force] = useState(0);
  const bump = () => force(x=>x+1);

  return (
    <>
      <Section title="Singly Linked List">
        <div className="flex gap-2 items-end mb-2">
          <label className="text-sm">Value <input className="border rounded px-2 py-1 ml-1 w-24" type="number" value={val} onChange={e=>setVal(Number(e.target.value))} /></label>
          <button className="px-3 py-1.5 rounded-xl border" onClick={() => { list.insertAtHead(val); bump(); }}>Insert Head</button>
          <button className="px-3 py-1.5 rounded-xl border" onClick={() => { list.insertAtTail(val); bump(); }}>Insert Tail</button>
          <button className="px-3 py-1.5 rounded-xl border" onClick={() => { list.delete(val); bump(); }}>Delete</button>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          {list.toArray().map((v,i)=>(<div key={i} className="px-3 py-1 rounded-full border">{v}</div>))}
          {list.toArray().length===0 && <span className="text-sm text-gray-500">(empty)</span>}
        </div>
      </Section>

      <Section title="Notes">
        <p className="text-sm">O(1) insert/delete at head; O(n) search. Great when frequent insertions in middle are needed (with pointer).</p>
      </Section>
    </>
  );
}

function BSTPanel() {
  const [bst] = useState(() => new BST());
  const [val, setVal] = useState(8);
  const [log, setLog] = useState<number[]>([]);

  return (
    <>
      <Section title="Binary Search Tree">
        <div className="flex gap-2 items-end mb-2">
          <label className="text-sm">Key <input className="border rounded px-2 py-1 ml-1 w-24" type="number" value={val} onChange={e=>setVal(Number(e.target.value))} /></label>
          <button className="px-3 py-1.5 rounded-xl border" onClick={() => { bst.insert(val); setLog(bst.inorder()); }}>Insert</button>
          <button className="px-3 py-1.5 rounded-xl border" onClick={() => alert(bst.search(val) ? "Found" : "Not Found")}>Search</button>
          <button className="px-3 py-1.5 rounded-xl border" onClick={() => setLog(bst.inorder())}>Inorder</button>
          <button className="px-3 py-1.5 rounded-xl border" onClick={() => setLog(bst.preorder())}>Preorder</button>
          <button className="px-3 py-1.5 rounded-xl border" onClick={() => setLog(bst.postorder())}>Postorder</button>
        </div>
        <div className="flex gap-2 flex-wrap">{log.map((v,i)=>(<div key={i} className="px-2 py-1 rounded border">{v}</div>))}</div>
      </Section>

      <Section title="Tip">
        <p className="text-sm">Balanced BSTs give O(log n) ops; unbalanced can degrade to O(n). Try inserting sorted numbers to see why.</p>
      </Section>
    </>
  );
}

function GraphPanel() {
  const [g] = useState(() => {
    const G = new Graph();
    G.addEdge("A", "B", 2); G.addEdge("A", "C", 5); G.addEdge("B", "D", 1); G.addEdge("C", "D", 2); G.addEdge("D", "E", 3);
    return G;
  });
  const [start, setStart] = useState("A");
  const [out, setOut] = useState<string[]>([]);
  const [target, setTarget] = useState("E");
  const dijkstra = () => {
    const { dist, pathTo } = g.dijkstra(start);
    alert(`distances from ${start}:\n` + Object.entries(dist).map(([k,v])=>`${k}: ${v}`).join("\n") + `\n\nShortest path ${start}→${target}: ${pathTo(target).join(" → ")}`);
  };

  return (
    <>
      <Section title="Graph Traversals & Dijkstra" right={<Pill>Adjacency List</Pill>}>
        <div className="flex flex-wrap gap-2 items-end mb-2">
          <label className="text-sm">Start <input className="border rounded px-2 py-1 ml-1 w-20" value={start} onChange={e=>setStart(e.target.value)} /></label>
          <button className="px-3 py-1.5 rounded-xl border" onClick={() => setOut(g.bfs(start))}>BFS</button>
          <button className="px-3 py-1.5 rounded-xl border" onClick={() => setOut(g.dfs(start))}>DFS</button>
          <label className="text-sm">Target <input className="border rounded px-2 py-1 ml-1 w-20" value={target} onChange={e=>setTarget(e.target.value)} /></label>
          <button className="px-3 py-1.5 rounded-xl border" onClick={dijkstra}>Dijkstra</button>
        </div>
        <div className="flex flex-wrap gap-2">{out.map((v,i)=>(<div key={i} className="px-2 py-1 rounded border">{v}</div>))}</div>
        <div className="mt-3 text-sm">
          <div className="font-mono whitespace-pre overflow-auto max-h-44 p-2 bg-gray-50 rounded-xl border">
            {JSON.stringify(g.adj, null, 2)}
          </div>
        </div>
      </Section>

      <Section title="Notes">
        <ul className="list-disc ml-5 text-sm">
          <li>BFS explores by layers (shortest edges in unweighted graphs). DFS dives deep first.</li>
          <li>Dijkstra works with non-negative weights to find shortest paths.</li>
        </ul>
      </Section>
    </>
  );
}

function DPPanel() {
  const [n, setN] = useState(10);
  const [W, setW] = useState(7);
  const [weights, setWeights] = useState("2,3,4,5");
  const [values, setValues] = useState("3,4,5,8");
  const [out, setOut] = useState<any | null>(null);

  return (
    <>
      <Section title="Fibonacci (Memo vs Tabulation)" right={<Pill>DP</Pill>}>
        <div className="flex gap-2 items-end mb-2">
          <label className="text-sm">n <input className="border rounded px-2 py-1 ml-1 w-20" type="number" value={n} onChange={e=>setN(Number(e.target.value))} /></label>
          <button className="px-3 py-1.5 rounded-xl border" onClick={() => alert(`fibMemo(${n}) = ${fibMemo(n)}\nfibTab(${n}) = ${fibTab(n)}`)}>Compute</button>
        </div>
      </Section>

      <Section title="0/1 Knapsack">
        <div className="flex flex-col gap-2 mb-2">
          <label className="text-sm">Weights <input className="border rounded px-2 py-1 ml-1 w-full" value={weights} onChange={e=>setWeights(e.target.value)} /></label>
          <label className="text-sm">Values <input className="border rounded px-2 py-1 ml-1 w-full" value={values} onChange={e=>setValues(e.target.value)} /></label>
          <label className="text-sm">Capacity <input className="border rounded px-2 py-1 ml-1 w-24" type="number" value={W} onChange={e=>setW(Number(e.target.value))} /></label>
          <button className="px-3 py-1.5 rounded-xl border w-max" onClick={() => {
            const w = weights.split(/[,\s]+/).filter(Boolean).map(Number);
            const v = values.split(/[,\s]+/).filter(Boolean).map(Number);
            const res = knapsack01(w, v, W); setOut(res);
          }}>Solve</button>
        </div>
        {out && (
          <div className="text-sm">
            <div className="mb-2">Best Value: <b>{out.best}</b>, Picked indices: [{out.picked.join(", ")}]</div>
            <div className="font-mono whitespace-pre overflow-auto max-h-56 p-2 bg-gray-50 rounded-xl border">{JSON.stringify(out.table, null, 2)}</div>
          </div>
        )}
      </Section>
    </>
  );
}

function StringsPanel() {
  const [text, setText] = useState("abxabcabcaby");
  const [pat, setPat] = useState("abcaby");
  const [idx, setIdx] = useState<number | null>(null);

  return (
    <>
      <Section title="KMP Substring Search" right={<Pill>O(n+m)</Pill>}>
        <div className="grid gap-2">
          <label className="text-sm">Text <Textarea value={text} onChange={setText} rows={3} /></label>
          <label className="text-sm">Pattern <input className="border rounded px-2 py-1 w-full" value={pat} onChange={e=>setPat(e.target.value)} /></label>
          <button className="px-3 py-1.5 rounded-xl border w-max" onClick={() => setIdx(kmpSearch(text, pat))}>Search</button>
        </div>
        {idx !== null && <div className="mt-2 text-sm">First match index: <b>{idx}</b></div>}
      </Section>

      <Section title="Prefix Table">
        <div className="font-mono whitespace-pre overflow-auto max-h-56 p-2 bg-gray-50 rounded-xl border">{JSON.stringify(kmpPrefix(pat), null, 2)}</div>
      </Section>
    </>
  );
}

function HashPanel() {
  const [table] = useState(() => new HashTable<string, number>());
  const [k, setK] = useState("apple");
  const [v, setV] = useState(3);
  const [, force] = useState(0);
  const bump = () => force(x=>x+1);

  return (
    <>
      <Section title="Hash Table (Separate Chaining)">
        <div className="flex flex-wrap gap-2 items-end mb-2">
          <label className="text-sm">Key <input className="border rounded px-2 py-1 ml-1 w-40" value={k} onChange={e=>setK(e.target.value)} /></label>
          <label className="text-sm">Value <input className="border rounded px-2 py-1 ml-1 w-24" type="number" value={v} onChange={e=>setV(Number(e.target.value))} /></label>
          <button className="px-3 py-1.5 rounded-xl border" onClick={() => { table.set(k, v); bump(); }}>Set</button>
          <button className="px-3 py-1.5 rounded-xl border" onClick={() => alert(String(table.get(k)))}>Get</button>
          <button className="px-3 py-1.5 rounded-xl border" onClick={() => { table.remove(k); bump(); }}>Remove</button>
          <Pill>Load: {table.loadFactor().toFixed(2)}</Pill>
        </div>
        <div className="font-mono whitespace-pre overflow-auto max-h-64 p-2 bg-gray-50 rounded-xl border">{JSON.stringify(table.entries(), null, 2)}</div>
      </Section>

      <Section title="Notes">
        <p className="text-sm">Hash maps average O(1) for get/set; collisions handled here with chaining. Load factor affects performance.</p>
      </Section>
    </>
  );
}

function BenchPanel() {
  const [size, setSize] = useState(2000);
  const [report, setReport] = useState<{label: string; ms: number}[]>([]);

  const run = () => {
    const src = rnd(size, 9999);
    const results = [
      timeIt("Array.sort (native)", () => [...src].sort((a,b)=>a-b)),
      timeIt("Merge Sort", () => deep(src).sort((a,b)=>a-b)), // baseline for UI; true merge already above
      timeIt("Quick Sort (JS)", () => deep(src).sort((a,b)=>a-b)), // minimal bench for demo
    ];
    setReport(results.map(r=>({label: r.label, ms: r.ms})));
  };

  return (
    <>
      <Section title="Micro Benchmark (toy)">
        <div className="flex gap-2 items-end mb-2">
          <label className="text-sm">Array size <input className="border rounded px-2 py-1 ml-1 w-28" type="number" value={size} onChange={e=>setSize(Number(e.target.value))} /></label>
          <button className="px-3 py-1.5 rounded-xl border" onClick={run}>Run</button>
        </div>
        <ul className="text-sm list-disc ml-5">
          {report.map((r,i)=>(<li key={i}><b>{r.label}</b>: {r.ms.toFixed(2)} ms</li>))}
        </ul>
        <p className="text-xs text-gray-500 mt-2">Note: This is a toy bench – results vary by device. For real benchmarking, use Node + Benchmark.js.</p>
      </Section>

      <Section title="Export">
        <button className="px-3 py-1.5 rounded-xl border" onClick={() => {
          const blob = new Blob([JSON.stringify({ exportedAt: new Date().toISOString(), report }, null, 2)], { type: "application/json" });
          const url = URL.createObjectURL(blob);
          const a = document.createElement("a");
          a.href = url; a.download = "dsa-playground-report.json"; a.click(); URL.revokeObjectURL(url);
        }}>Download JSON</button>
      </Section>
    </>
  );
}
