## PiPNN: Ultra-Scalable Graph-Based

## Nearest Neighbor Indexing

## Tobias Rubel

#### UMD

#### College Park, Maryland

#### trubel@umd.edu

## Richard Wen

#### UMD

#### College Park, Maryland

#### rwen@umd.edu

## Laxman Dhulipala

#### UMD and Google Research

#### College Park, Maryland

#### laxman@umd.edu

## Lars Gottesbüren

#### Google Research

#### Zürich, Switzerland

## Rajesh Jayaram

#### Google Research

#### New York City, New York

## Jakub Łącki

#### Google Research

#### New York City, New York

### ABSTRACT

```
The fastest indexes for Approximate Nearest Neighbor Search
(ANNS) today are also the slowest to build: graph-based methods
like HNSW and Vamana achieve state-of-the-art query performance
but have prohibitively large construction times due to relying on
random-access-heavy beam searches. In this paper, we introduce
PiPNN (Pick-in-Partitions Nearest Neighbors), an ultra-scalable
graph construction algorithm that avoids this “search bottleneck”
that existing graph-based methods suffer from.
PiPNN’s core innovation is HashPrune, a novel online pruning
algorithm which dynamically maintains sparse collections of edges.
HashPrune enables PiPNN to partition the dataset into overlapping
sub-problems, efficiently perform bulk distance comparisons via
dense matrix multiplication kernels, and stream a subset of the
edges into HashPrune. HashPrune guarantees bounded memory
during index construction which permits PiPNN to build higher
quality indices without the use of extra intermediate memory.
Our extensive experimental study demonstrates that PiPNN
builds state-of-the-art indexes up to 11. 6 ×faster than Vamana
(DiskANN) and up to 12. 9 ×faster than HNSW. We show that these
improvements extend to downstream tasks, yielding speedups of
up to 1. 9 ×for approximate𝑘-NN graph construction. PiPNN is
significantly more scalable than recent algorithms for fast graph
construction. PiPNN builds indexes at least 19. 1 ×faster than MI-
RAGE and 17. 3 ×than FastKCNA while producing indexes that
achieve significantly higher query throughput. PiPNN enables us to
build, for the first time, high-quality ANN indexes on billion-scale
datasets in under 20 minutes using a single multicore machine.
```
### 1 INTRODUCTION

```
High-dimensional vector embeddings are a fundamental datatype
used in modern search, information retrieval, classification, and
recommendation applications. For example, vector embeddings are
the backbone of a diverse set of applications including entity reso-
lution [ 26 , 44 ], retrieval-augmented generation (RAG) systems for
generating context for large language models (LLMs) [ 16 , 25 ], and
recommendation systems [ 29 , 34 , 40 ]. A fundamental problem in
these applications is nearest neighbor search: given a query vector
(point), find its nearest neighbors from a large set of points accord-
ing to some distance function (e.g., 𝐿 2 or inner product distance).
```
```
OpenAI
Wikipedia Cohere
```
```
BIGANN-1BMSSPACEV-1BMSTURING-1BDEEP-1B
```
```
0
```
```
2
```
```
4
```
```
6
```
```
8
```
```
10
```
```
12
```
```
Speedup over HNSW
```
```
PiPNN (1 Replica) HCNNG Vamana (1 Pass) HNSW
```
```
Figure 1: Build time speedup compared to HNSW on six bench-
marks, including billion-scale inputs from big-ann-benchmarks.
Due to the notorious difficulty of finding the exact nearest neigh-
bors in high-dimensional spaces [ 10 ] and the fact that real-world ap-
plications typically tolerate small errors, modern embedding-based
applications leverage approximate nearest-neighbor search (ANNS).
In recent years, graph-based indexing methods such as HNSW [ 27 ],
Vamana [ 22 , 24 ], and NSG [ 15 ] have become the standard for high-
recall and low-latency ANNS. These methods add edges between
nearby points along with carefully chosen long-range edges for
navigability, enabling fast queries using beam search.
Although graph-based indexing methods are the fastest for query-
ing, they are also extremely slow to build, and their high construction
cost undermines their utility in real-world applications where index
build time is just as important as query performance. This occurs,
for example, in applications where the index must be rebuilt peri-
odically, or the index parameters must be tuned for new datasets.
Moreover, ANNS often functions as a subroutine within broader
algorithmic frameworks, including accelerated𝑘-means [ 38 ], hi-
erarchical clustering [ 14 , 43 ], and approximate𝑘-NN graph con-
struction [ 11 ]. It is also central to constructing high-dimensional
objects like spanners, minimum spanning trees, and single-linkage
clustering [ 2 , 5 , 8 , 9 ]. In these contexts, the index construction time
counts directly toward the primary metric to optimize, namely end-
to-end execution time. Lastly, the slow construction time of existing
graph-based methods presents a severe limitation for large-scale
use cases that must index billions of points.
The root cause is that existing graph-based methods rely on
incremental insertion, where each point requires a beam search
```
# arXiv:2602.21247v2 [cs.DB] 24 May 2026


```
Tobias Rubel, Richard Wen, Laxman Dhulipala, Lars Gottesbüren, Rajesh Jayaram, and Jakub Łącki
```
```
on the partial index to find its neighbors. This incremental con-
struction approach has been shown to produce very high quality
graphs, and has been used in all state-of-the-art graph-based in-
dexes, including HNSW [ 27 ] and Vamana [ 22 ]. However, the use
of beam search during construction leads to a significant volume
of random memory accesses, leading to poor pipelining and fre-
quent cache misses. As a result, incremental graph-based indices
suffer from the search bottleneck where the majority of time spent
during index construction is spent performing beam searches, lead-
ing to construction times on the order of many hours (and even
days) for billion-scale datasets, even on many-core, high-bandwidth
machines. This situation presents the following open problem:
```
```
Can we obtain state-of-the-art index quality search graphs
while circumventing the search bottleneck inherent to
existing graph-based nearest neighbor search algorithms?
```
In this paper we resolve this problem by departing significantly
from the approach used by previous state-of-the-art graph indexing
methods. We instead combine a novel algorithm for maintaining
high-quality sparse adjacency lists across batches of candidate edge
insertions (HashPrune) with partitioning-based approaches to
efficiently identify candidate neighbors. Our graph indexing algo-
rithm is called PiPNN (for Pick-in-Partitions Nearest Neighbors,
pronounced ‘Pippin’), a new ultra-scalable algorithm for building
ANNS indexes based on (1) partitioning the underlying points into
a collection of small-sized overlapping partitions called leaves via
dense ball carving, (2) building sparse nearest neighbor graphs
within each leaf, so as to produce potential candidate edges for
the search graph, and (3) leveraging an online pruning algorithm
(HashPrune) to balance both the nearness of edges in the graph
as well as directional diversity.
PiPNN circumvents the search bottleneck by eliminating search
from the graph-building process altogether. Instead, we design
PiPNN so most of its work is spent in computing all-pairs distances
within the leaf partitions using highly optimized general matrix
multiply (GEMM) operations using the Eigen library [ 20 ]. This
algorithmic feature of PiPNN enables us to leverage extremely op-
timized kernels for matrix-matrix multiplication that benefit from
vectorized instructions, and are available on essentially all mod-
ern hardware platforms. As we illustrate in Figure 1, compared to
popular (and at this point, highly optimized) methods like Vamana
and HNSW that suffer from the search bottleneck, PiPNN is able to
construct state-of-the-art indexes up to 11.6x faster than Vamana
(average 6.32x) and up to 12.9x faster than HNSW (average 10.4x).
Compared to recent work on improving the scalability of graph-
based index construction, including MIRAGE [ 39 ], FastKCNA [ 41 ],
and LSH-APG [ 45 ], PiPNN achieves even greater speedups. We note
that these methods were primarily evaluated on 1-million scale
datasets. By contrast, we provide a thorough evaluation of PiPNN
at both the 100 million- and billion-scale and show significant
speedups over all prior works, which is precisely the regime where
scalable index construction is most critical.
In summary, the main contributions of this paper are:
(1)A novel online and history-independent pruning algorithm
called HashPrune capable of yielding high-quality sparse search
graphs when given noisy candidate lists.

```
(2)The PiPNN algorithm that quickly approximates topological
neighborhoods via overlapping partitioning, along with fast
methods for picking batches of candidate edges from the clus-
ters for consideration by HashPrune.
(3)Experimental results comparing PiPNN with existing state-of-
the-art graph-based methods with speedups of up to 11. 6 ×
for building graph indexes of identical quality and up to 5. 8 ×
across four billion-scale datasets.
```
### 2 PRELIMINARIES

```
Approximate Nearest Neighbor Search (ANNS) Background.
LetX ⊆R𝑑be a collection of𝑛vectors (points) in𝑑dimensions.
Consider some arbitrary dissimilarity measure ∥·,·∥ overR𝑑.
```
```
Definition 1. (k-nearest neighbor search) Given query point𝑞 ∈
R𝑑, the k nearest neighbor search overXproduces the setK ⊆ X
such that, max𝑝∈K∥𝑞,𝑝∥ ≤ min𝑝∈X\𝐾∥𝑞,𝑝∥.
```
```
In this work we consider the task of approximatingK. As is
typical, we evaluate the approximations using recall.
```
```
Definition 2. (k@k’ recall) LetKbe the set of𝑘nearest neighbors
of𝑞inXas defined in Definition 1. Then, givenK′⊆ Xof size𝑘′,
the k@k’ recall ofK′is|K∩K
```
```
′|
|K|.
In this paper we follow the literature (see [ 4 , 28 , 36 , 37 ]) in
focusing on 10@10 recall, and will use the term recall to denote
this specific notion. For a set𝑄of queries, we use recall to denote
the mean recall over 𝑄.
Graph Based Indexes. Given a set of pointsX, the family of graph
indexing methods build a navigation graph𝐺(X,𝐸), where each
point𝑝 ∈ Xis represented by a vertex in the graph. The goal is
to build the graph to ensure that it is searchable via beam search,
which we describe next.
Beam Search (Algorithm 1). Given a navigation graph over points
X, the standard query procedure is a greedy search with interme-
diate storage (called the beam). Pseudocode for this procedure is
given in Algorithm 1. Given a source node𝑠in the graph and a
query point𝑞, a beam search with beam width𝐿works as follows.
We maintain the two sets: a set𝐵containing the𝐿points seen
that are closest to𝑞(often called the “beam”), and the set𝑉of all
points visited. Visiting a point𝑝 ∈ 𝐵\𝑉is done by adding all its
unseen neighbors to the beam, then adding𝑝to the visited set. The
algorithm begins by initializing the beam with only the source𝑠, as
well as an empty visited set. Then, we visit the unvisited node in
the beam that is closest to𝑞, repeating until every node in the beam
has been visited. Because the distance of the points in the beam
to the target decreases monotonically, this process will eventually
converge, at which point we return the final state of the beam.
```
### 2.1 Offline Pruning & Incremental Builds

```
Most state-of-the-art graph-based ANNS methods, including HNSW
[ 27 ], Vamana [ 22 ], and NSG [ 15 ], can be characterized as employ-
ing two-stage process for each point: (1) identifying a subset of
candidate neighbors from the universe of points, and (2) applying
a pruning algorithm on those candidates to select a sparse set of
edges that enforces directional diversity.
```

```
PiPNN : A Framework for Ultra-Scalable Graph-Based Nearest Neighbor Indexing
```
Algorithm 1: BeamSearch
Input: 𝐺=(X,𝐸) a graph over set of pointsX fitted with
dissimilarity measure∥·,·∥,𝑠 ∈Xa source point,𝑞a target
point, 𝐿 ∈N the beam width parameter
Output: 𝐵 ⊆ X the 𝐿 closest points to 𝑞 encountered during the
search
1 𝐵 ←{𝑠};
2 𝑉 ←∅;
3 while 𝐵\𝑉≠∅ do
4 𝑝 ← argmin𝑢∈𝐵\𝑉∥𝑢,𝑞∥;
5 𝐵 ← 𝐵∪ Neighbors(curr);
6 𝑉 ← 𝑉 ∪{𝑐𝑢𝑟𝑟};
7 if |𝐵|> 𝐿 then
8 𝐵 ← closest 𝐿 points to 𝑞 in 𝐵;

9 return 𝐵;

```
Pruning algorithms are typically derived from the construction
method of the Relative Neighborhood Graph (RNG), which selects
edges based on the proximity of candidates to one another [ 3 ].
Insofar as pruning requires doing pairwise comparisons between
candidate neighbors, these pruning rules are inherently quadratic in
the number of candidates. Consequently, one cannot simply prune
the entire universe of𝑛points for every vertex, as the resulting
𝑂(𝑛^3 ) construction time would be prohibitive for large datasets.
Many methods circumvent this issue by way of an incremental
construction paradigm: for each point being inserted, the algorithm
performs a search on the current (partial) index to identify a local
collection of candidate neighbors. These candidates are then re-
fined using a pruning kernel, such as the RobustPrune algorithm
used in Vamana [ 22 ] (see Algorithm 2) or the RNG-based rules in
HNSW and NSG[ 15 , 27 ]. While effective at identifying relevant
edges, incremental construction suffers from the search bottleneck,
where the majority of build time is spent on latency-bound random
memory accesses that fail to exploit modern hardware parallelism.
Recent attempts to achieve faster index construction try to reduce
the cost of these searches. MIRAGE [ 39 ] aims to improve the qual-
ity of the base graph in order to make the searches more efficient.
LSH-APG [ 45 ] uses LSH to avoid some distance computations in
the search, as well as to pick a closer starting point. FastKCNA [ 41 ]
introduces a ’refinement-before-search’ framework. Rather than
searching a partial graph to find candidates for pruning, it first ap-
plies a relaxed RNG constraint (𝛼-pruning) to an approximate KNN
graph, yielding faster searches for identifying candidate neighbors.
However, these methods do not eliminate searches entirely, and
thus still suffer from the associated inefficiencies.
An alternative to the search-based approach is to use some kind
of overlapping partitioning, as seen in methods like HCNNG [ 30 ].
Partitioning approaches identify potential neighbors without search-
ing the graph which allows for more cache-friendly algorithms via
batched distance computations. However, the curse of dimension-
ality forces clustering methods to generate a significantly larger
universe of candidate points per vertex than search-based meth-
ods. Storing all these candidates for a final “offline” prune would
require a massive memory footprint, and the resulting candidate
lists would be prohibitively large to prune using existing𝑂(𝑛^2 )
```
```
pruning methods. Moreover, existing partitioning approaches do
not consistently produce state of the art ANNS indexes. This is
because, absent a viable pruning strategy, HCNNG combines edges
produced in each cluster via union, which yields dense adjacency
lists with directional redundancy.
The ideal solution for a partitioning algorithm would be to prune
candidates as they are discovered in each partition. However, ex-
isting pruning kernels do not lend themselves to online, batched
execution. Firstly, each batch of candidates would require a qua-
dratic number of distance computations over the union of the batch
and existing partial adjacency list, which is computationally prohib-
itive. Moreover, if a candidate list is divided into arbitrary batches
(e.g., partition by partition), traditional pruning rules lose global
coherence. For example, a point𝑧might be pruned in an early batch
because of an "intermediary" point𝑦. If𝑦is itself later pruned dur-
ing the processing of another partition, the algorithm no longer
has a justification for the pruning of𝑧, potentially leading to a
fragmented graph structure. Thus, existing pruning methods lack
the history-independence required to ensure that the final graph
structure remains consistent regardless of the order that candi-
dates are processed. These challenges motivate our development
of HashPrune, which maintains a deterministic and sparse graph
across arbitrary collections of candidates while simultaneously
guaranteeing a bounded memory footprint and fast updates.
```
```
Algorithm 2: RobustPrune
Input:N a collection of candidate points fitted with dissimilarity
measure∥·,·∥, points 𝑥 ∈X, 𝛼 ∈R, 𝑅 ∈N
Output: 𝐸 ⊆ X×X
1 whileN≠∅ and|𝐸|< 𝑅 do
2 𝑦 ← argmin𝑦∈X∥𝑥,𝑦∥;
3 𝐸.Add((𝑥,𝑦));
4 N .Remove(𝑦);
5 for 𝑧 ∈N do
6 if 𝛼 ·∥𝑦,𝑧∥< ∥𝑥,𝑧∥ then
7 N .Remove(𝑧)
```
```
8 return 𝐸;
```
### 3 ONLINE PRUNING VIA HASHPRUNE

```
In this section, we devise a history-independent online pruning
technique which we call HashPrune which is integral to the
partitioning approach used in PiPNN.
Motivation. As mentioned in Section 2.1, pruning algorithms
ensure that each vertex connects to neighbors in many different di-
rections so that greedy search can quickly route from any one point
to any other. HashPrune accomplishes this objective via locality-
sensitive hashing (LSH) [ 12 ] to prune candidate edges which are
likely to be directionally similar to other candidates. The use of LSH
also allows for the use of low dimensional sketches during pruning,
which eliminates the need to fetch high dimensional vectors. We
give pseudocode for the HashPrune algorithm in Algorithm 3, and
formally describe the components of HashPrune next.
Residualized Hashing. HashPrune is initialized by generating
a set of𝑚random hyperplanes{H 1 ,.. .,H𝑚}through the origin.
```

```
Tobias Rubel, Richard Wen, Laxman Dhulipala, Lars Gottesbüren, Rajesh Jayaram, and Jakub Łącki
```
```
p
```
```
𝑐 1
𝑐 2
```
```
𝑐 3
```
```
Added (𝑐′)
```
## ×Pruned

```
(a) Smaller𝑚
```
```
p
```
```
𝑐 1
𝑐 2
```
```
𝑐 3
```
```
Added (𝑐′)
```
```
(b) Larger𝑚
```
Figure 2: Impact of resolution on neighbor retention. Shading
represents the probability of collision based on number of bits used
in the HashPrune hash (𝑚). (a) At coarse resolution,𝑐′collides
with and evicts the farther neighbor𝑐 2. (b) At finer resolution, both
edges are retained.

For each point𝑝, we generate an individualized hash functionℎ𝑝,
which takes a candidate point𝑐and hashes the residual of𝑝and𝑐:

##### ℎ𝑝(𝑐)=

##### Ê𝑚

```
𝑖= 1
```
##### (

```
1 ifH𝑖·(𝑐− 𝑝) ≥ 0 ,
0 ifH𝑖·(𝑐− 𝑝)< 0.
```
##### (1)

```
Here⊕designates the concatenation operation. That is, we pro-
duce the𝑖-th bit inℎ𝑝by checking which side of the𝑖-th hyperplane
the residual falls on.
Insertion Procedure. To use HashPrune to prune candidate
neighbors for a given point𝑝, we initialize a reservoir𝑀of size
ℓ𝑚𝑎𝑥(Line 1). To insert a candidate neighbor𝑐, we compute the
individualized hash functionℎ𝑝(𝑐). If another candidate with the
same hash already exists in the reservoir𝑀(Line 3), then we keep
the candidate that is closer to𝑝(Line 4). Otherwise, we insert the
incoming candidate into the reservoir, and if the reservoir is full
(Line 10), we evict the existing candidate that is furthest from𝑝
(Line 12). Our next theorem shows that HashPrune can be easily
used in settings where candidate neighbors of points are obtained
in arbitrary orders due to its history-independent guarantees.
```
```
Theorem 3.1. HashPrune is history-independent: Given a hash
functionℎ𝑝for some point𝑝, a collection of candidates𝐶, and a
reservoir sizeℓ, the final adjacency list produced by HashPrune is
unique and independent of the insertion order of candidates in𝐶.
```
See Supplement A.2 for a detailed proof of this result.
Why HashPrune Works. For each𝑝, and candidates𝑐,𝑐′, the
classic theorem for LSH gives us that𝑃[ℎ𝑝(𝑐)=ℎ𝑝(𝑐′)]=( 1 −𝜃𝜋)𝑚,
where𝜃is the angle between𝑐−𝑝and𝑐′−𝑝[ 12 ]. Thus, HashPrune
draws probabilistic cones centered on each𝑝, and retains the nearest
point in each cone. Figure 2 illustrates how the density of the final
graph is impacted by the number of hyperplanes used for the hashes.
As opposed to prior pruning algorithms which require𝑂(ℓ)com-
parisons to insert a new point into a pruned adjacency list of sizeℓ,
HashPrune can be implemented to both prune and insert a new
candidate in𝑂(logℓ)comparisons so long as the reservoir is not
full. When the reservoir is full (i.e.ℓ= ℓ𝑚𝑎𝑥), HashPrune requires
𝑂(ℓ𝑚𝑎𝑥)time to insert a candidate when there is no collision, but
otherwise retains the𝑂(logℓ)behavior for insertion. The linear

```
insertion time for non-colliding candidates in a full reservoir results
from having to update the furthest neighbor in the reservoir.
```
```
Algorithm 3: HashPrune
Input:V a stream of candidate points, fitted with dissimilarity
measure∥·,·∥, point 𝑝, size limit ℓ𝑚𝑎𝑥∈N
Output: A set𝐶 ⊆ V of size min(|V|,ℓ𝑚𝑎𝑥)
1 𝑀 ← an empty key-value store mapping hashes to points;
2 for 𝑐 ∈V do
3 if 𝑀.HasKey(ℎ𝑝(𝑐)) then
4 if ∥𝑝,𝑐∥< ∥𝑝,𝑀.Get(ℎ𝑝(𝑐))∥ then
5 𝑀 .Set(ℎ𝑝(𝑐),𝑐);
```
```
6 else
7 if |𝑀|< ℓ𝑚𝑎𝑥then
8 𝑀 .Set(ℎ𝑝(𝑐),𝑐);
9 else
10 𝑧= argmax𝑦∈𝑀∥𝑝,𝑦∥;
11 if ∥𝑝,𝑐∥< 𝑧 then
12 𝑀 .Remove(𝑧);
13 𝑀 .Set(ℎ𝑝(𝑐),𝑐);
```
```
14 return 𝑀 .Values();
```
```
Implementation. For each point𝑝, we implement its correspond-
ing key-value store𝑀as an array of capacityℓ𝑚𝑎𝑥, in which each
candidate𝑐in𝑀is represented by its 4-byte point ID, the 2-byte
hashℎ𝑝(𝑐), and the norm∥𝑝,𝑐∥stored as a 2-byte floating point
number (bf16). Thus, each slot in the reservoir uses a mere 8 bytes.
This compactness is important to maintaining a reasonable memory
footprint, as we keep one reservoir per point.
Instead of using the actual points𝑝,𝑐to compute the residual
𝑐−𝑝, we pre-compute𝑚-dimensional sketches for each point in the
dataset by letting𝑆𝑘𝑒𝑡𝑐ℎ(𝑣)=[𝑣·H𝑖: 0≤ 𝑖< 𝑚]. Thenℎ𝑝(𝑐)can
be computed by concatenating the sign of the difference between
each dimension of𝑆𝑘𝑒𝑡𝑐ℎ(𝑐)and𝑆𝑘𝑒𝑡𝑐ℎ(𝑝). This yields the same
hashes but requires loading𝑚-dimensional vectors which are one
or two orders of magnitude smaller than the𝑑-dimensional vectors
PiPNN is designed to index.
Candidates in the reservoir are stored in sorted order by hash,
letting us quickly binary search for a given hash. When the reservoir
is full, we identify the farthest candidate𝑧= argmax𝑦∈𝑀∥𝑝,𝑦∥in
the reservoir through a simple linear scan, then subsequently cache
it. If we find that a new candidate does not warrant an eviction
(because its norm is greater than that of𝑧), then we can reuse the
cached 𝑧 for the next candidate without a linear scan.
Although one could likely implement a more efficient data struc-
ture for the key-value store (e.g. a compact hash table), profiling
indicates that maintaining the reservoir typically constitutes no
more than 5% of our total construction time. Thus, unless signifi-
cant improvements are made to the speed of the rest of the PiPNN
algorithm, improving the reservoir scheme is unlikely to yield a
notable impact on overall running time.
```

```
PiPNN : A Framework for Ultra-Scalable Graph-Based Nearest Neighbor Indexing
```
### 4 PIPNN ALGORITHM

```
Motivation. PiPNN uses HashPrune in conjunction with highly
optimized partitioning-based approaches to index construction to
achieve state of the art index quality in line with that of the best
incremental methods. PiPNN takes better advantage of modern
hardware by 1) partitioning the points into cache-friendly units
of work and 2) once vector embeddings are grouped into cache-
friendly chunks, compute all pairwise distances within the chunk
using highly optimized matrix multiplication kernels.
Overview of the PiPNN Algorithm. High-level pseudo-code for
PiPNN is given in Algorithm 4. At a high level PiPNN works as
follows: (1) creating an overlapping partitioningB={𝑏 1 ,.. .,𝑏𝑡}
of the points inX(Line 2), (2) generating candidate edges between
each point and some subset of the points it resides with via a simple
sparsification routine (Line 4), and (3) using HashPrune to prune
those candidate edges for each point (Line 5). We are able to improve
query performance further by performing a final RobustPrune of
the overall candidate list for each 𝑥 ∈X (Line 8).
```
Algorithm 4: PiPNN
Input:X a collection of points with dissimilarity measure, 𝐵𝑃 a
collection of hyper-parameters
Output: 𝐺=(𝑉,𝐸) a graph overX
1 G← (X,∅);
2 B← Partition(X, 𝐵𝑃);
3 for 𝑏𝑖∈ B do in parallel
4 new_edges← Pick(𝑏𝑖, BP);
5 G.Prune_And_Add_Edges(new_edges);

6 if 𝐵𝑃.final_prune then
7 for 𝑥 ∈X do in parallel
8 PruneNode(x);

9 return 𝐺 ;

### 4.1 Partitioning via Randomized Ball Carving

```
As previously mentioned, incremental methods seek to find for
each point a collection of nearby points by way of greedy searches
on the partial index. PiPNN instead seeks to find an overlapping
partitioning of the dataset such that nearby points are likely to
fall into the same partition. Possible solutions to this problem in-
clude locality-sensitive hashing, k-means partitioning, and repeated
binary partitioning [ 7 , 21 , 30 ]. We include an ablation study in Sup-
plement A.1 which evaluates these choices, but ultimately found
that Randomized Ball Carving (RBC) achieved the best balance of
construction time and index quality.
In a subproblemP ⊆ X, Randomized Ball Carving (RBC) selects
ℓrandomly chosen points𝑝 1 ,.. .,𝑝ℓ∈ Pto serve as leaders. It then
assigns each point inPto its nearest leader, formingℓsubproblems.
In PiPNN, we letℓbe the smaller of some percentage of the|P|
points in a subproblemP𝑠𝑎𝑚𝑝·|P|and a hard-cap (typically 1000).
It then recursively performs ball carving on subproblems with more
than𝐶𝑚𝑎𝑥points, until eventually producing leavesB 1 ,.. .,B𝑏each
of size at most𝐶𝑚𝑎𝑥. Note that a single execution of RBC yields a
disjoint partition. This procedure could be run multiple times to
```
```
produce a collection of partitions which jointly produce an over-
lapping partitioning, in a process we refer to as replication. In the
following, we propose a more computationally efficient approach.
Fanout. Given that the recursion tree in RBC has arity roughly
P𝑠𝑎𝑚𝑝· 𝑛, we can avoid replicating the partitioning process and
instead assign each𝑥to the subproblems associated with its𝑘near-
est leaders at the top level. The arity of the tree is large at high
levels. Thus, a single recursive process can be used to produce an
overlapping clustering which is comparable with respect to its util-
ity in index construction (see Supplemental Figure 8). This fanout
optimization yields performance improvements because it replaces
the process of replicating the entire recursive process repeatedly
(each of which requires a pass over the data for each replicate) with
a single recursive process and, therefore, a single pass over the
entire dataset at the top level (Supplemental Sec. A.1.4).
While this modification improves index construction times, we
found that delaying some fanout to lower recursion levels addition-
ally aids performance, since the subproblems are smaller (yielding
better cache-behavior), the top-k aggregation is cheaper, and the
parallel group-by operation to form clusters runs on fewer entries.
We observed that fanout along a roughly geometric sequence (e.g.,
10 on the top level, 3 on the second level) produces neighbor lists of
similar quality with extremely fast partitioning times. We refer to
this scheme as multi-level fanout. This scheme can be used to re-
place replicating the entire partitioning procedure with no change
in index performance (Supplemental Figure 8), but improves parti-
tioning time by between 1. 35 × and 2. 5 × (as seen in Figure 3).
To ensure that the partitions are all within bounded sizes which
will likely fit into cache during the prune step we define the user
tunable maximum cluster size𝐶𝑚𝑎𝑥to be fairly small (typically be-
tween 1024–2048), and return a partition once it is beneath this
size. We also define a minimum cluster size𝐶𝑚𝑖𝑛and merge parti-
tions randomly (never exceeding a cluster size of𝐶𝑚𝑎𝑥) if they fall
beneath this cutoff. This ensures that each partition gives enough
potential neighbors for the prune algorithm. Algorithm 5 gives
pseudocode for this partitioning strategy.
```
### 4.2 Leaf Building

```
RBC produces an overlapping partitioning,B={𝑏 1 ,.. .,𝑏𝑡}where
each𝑏𝑖is a collection of up to𝐶𝑚𝑎𝑥points we term “leaves”. We can
obtain distances within a given leaf more efficiently by precomput-
ing a distance matrix small enough to fit in cache, something that
would be infeasible on the full dataset and without the fact that we
restrict each point’s interactions to those other points which share
a cluster with it. This allows us to use efficient matrix multiplication
kernels [ 20 ] to compute all-pairs-distances and thus amortize the
cost of distance computations within a cluster.
While at this point it is possible to simply use HashPrune to
prune the entirety of the leaf for each point, performance can be
improved by first carefully picking a sparser set of candidate edges
to prune. Considering too many candidates can result in indexes of
excessive density which perform poorly in practice, in addition to
adding needless computation which slows down indexing time. This
is not a defect unique to HashPrune: In our testing we found both
RobustPrune and the RNG prune condition produced degraded
indexes when provided with massive candidate lists.
```

```
Tobias Rubel, Richard Wen, Laxman Dhulipala, Lars Gottesbüren, Rajesh Jayaram, and Jakub Łącki
```
```
Algorithm 5: Randomized Ball Carving
⊲Constants:𝐶𝑚𝑎𝑥(max cluster size),𝐶𝑚𝑖𝑛(min cluster size),P𝑠𝑎𝑚𝑝
(leader fraction), fanout :Z→Z (function of depth )
Input:P ⊆ X a collection of points with dissimilarity measure,
depth= 0
Output: non-disjoint partitionB={𝑏 1 ,.. .,𝑏𝑡}
1 if |P| ≤ 𝐶𝑚𝑎𝑥then
2 return {P};
3 L ← SampleLeaders(P,P𝑠𝑎𝑚𝑝·|P|);
4 local_fanout← min(fanout(depth),|P|);
5 B ←{∅,.. .,∅} of sizeL ;
6 for each point 𝑥 ∈ P do in parallel
7 nearest_leaders← the local_fanout leaders inL closest to 𝑥 ;
8 for each leader 𝑙 ∈ nearest_leaders do
9 Add 𝑥 to the set inB corresponding to 𝑙 ;
```
10 Merge(B,𝐶𝑚𝑖𝑛,𝐶𝑚𝑎𝑥);
11 B’←∅;
12 for 𝑏 ∈ B : |𝑏| ≥ 𝐶𝑚𝑎𝑥do in parallel
13 B.𝑟𝑒𝑚𝑜𝑣𝑒(𝑏);
14 B ←B∪ Partition(𝑏,depth+1);

15 returnB;

```
Instead, we pick a subset of the possible edges to add as can-
didates to the HashPrune structure for each point by building a
k-NN graph over the leaf and then bi-directing the edges. We con-
sidered other methods as well, including minimum spanning trees,
directed k-NNs, inverted k-NNs, and running RobustPrune for
each point within the leaf. More detailed descriptions of these other
methods are available in Supplement A.3, along with an ablation of
their relative performances.
```
### 4.3 Final Pruning

```
HashPrune can be used alone to produce high quality indexes by
simply taking the graph generated by building k-NN graphs on
leaves produced by randomized ball carving. PiPNN also provides
the option to post-process the graph with a final RobustPrune on
each point. Because RobustPrune has a real-valued𝛼parameter
for tuning sparsity, it is more finely tunable to the topology of a
given dataset than HashPrune which uses discrete hash buckets
to estimate angular cones. This allows for additional sparsification,
which we find improves query throughput by a small amount. The
size of the adjacency lists provided to RobustPrune by PiPNN are
very small relative to those in Vamana^1 , and thus the cost of the
final pruning step is low.
```
### 5 EXPERIMENTAL EVALUATION

```
Machines. Our experiments were carried out on one of two iden-
tical machines with four Intel Xeon Platinum 8160 CPUs each (to-
taling 192 CPUs) and 1.5TB of DRAM. Our experiments on billion-
scale datasets and𝑘-NN graph building experiments were run on a
Google Cloud c4-highmem-288 machine with 288 vCPUs and 2.2TB
```
(^1) Our experiments with Vamana showed that due to adding back edges, RobustPrune
can be called on candidate lists exceeding 10k elements.
of DRAM. Results in a single plot or chart are always generated on
the same machine so as to obviate any differences.
Baseline Algorithms & Parameters. We compare our approach
against five graph-based algorithms. We evaluated PiPNN against
HCNNG [ 30 ], Vamana [ 22 ], and HNSW [ 27 ]. Both Vamana and
HNSW are widely used due to their excellent query performance.
We utilize the implementations of HCNNG and Vamana provided by
the ParlayANN library [ 28 ]. To the best of our knowledge, the Par-
layANN implementation of Vamana is the fastest and highest quality
implementation of a shared-memory graph-based ANNS algorithm
available today. We also evaluate PiPNN against MIRAGE [ 39 ] and
FastKCNA [ 41 ], which aim to achieve high-quality graphs in less
build time than Vamana and HNSW. We do not evaluate against
LSH-APG [ 45 ], which was shown in [ 41 ] to build indexes slower
than FastKCNA. Regarding configuration, we adopt the optimal
hyperparameters identified in ParlayANN for Vamana, while for
HNSW, HCNNG, MIRAGE, and FastKCNA we follow the hyperpa-
rameters recommended by their respective authors. To ensure a
fair comparison, we restrict the maximum graph degree to 64 for
all methods with the exception of HCNNG. For HCNNG, for which
the maximum degree is dependent on other parameters, we allow
a maximum degree of 90 as recommended by the authors. We were
unable to run MIRAGE and FastKCNA on billion-scale datasets be-
cause they exceeded the 2.2TB of DRAM available on our machine,
and additionally FastKCNA could not be run on WikiCohere due
to a lack of support for the MIPS dissimilarity measure.
Datasets. We evaluate on four billion-scale datasets and two
smaller high-dimensional datasets, accessed through the BigANN
Benchmarks framework [ 36 , 37 ]. Table 1 summarizes the datasets.
For ablation studies, we use 100-million sized subsets of BigANN
and MS-SPACEV, as well as Wikipedia-Cohere and OpenAI-ArXiv.
Table 1: Summary of datasets used in our experiments.
Dataset Size Dim. Type Metric Cite
BigANN 1B 128 uint8 L2 [23]
DEEP 1B 96 float32 L2 [6]
MS-SPACEV 1B 100 int8 L2 [37]
MS-Turing 1B 100 float32 L2 [37]
Wikipedia-Cohere 35M 768 float32 MIPS [33]
OpenAI-ArXiv 1M 1536 float32 L2 [31]

### 5.1 Optimizations in PiPNN

```
We first describe and evaluate the key algorithmic and implementa-
tion optimizations in PiPNN which jointly contribute to the superior
index construction times shown in Section 5.2.
Multi-Level Fanout. In Section 4.1, we explained that PiPNN
uses a single recursive process to select multiple leaders per level,
referred to as fanout, instead of repeatedly replicating the entire
partitioning. We claimed that this significantly speeds up parti-
tioning times. In particular, replication requires us to execute the
topmost levels of recursion many times across different runs. On
the other hand, fanout allows us to dispense many copies of each
point into lower subproblems from only a single execution at the
root level. Multi-level fanout lets us partially extend this benefit
past the root level to the second level or deeper. See Supplement
```

```
PiPNN : A Framework for Ultra-Scalable Graph-Based Nearest Neighbor Indexing
```
```
10 20 30 40 50 60
Combined Fanout (num clusters * fanout)
```
```
0
```
```
500
```
```
1000
```
```
1500
```
```
Graph Build Time (s)
```
```
SpaceV
```
```
10 20 30 40 50 60
Combined Fanout (num clusters * fanout)
```
```
0
```
```
50
```
```
100
```
```
150
```
```
Graph Build Time (s)
```
```
OpenAI
```
```
10 20 30 40 50 60
Combined Fanout (num clusters * fanout)
```
```
0
```
```
500
```
```
1000
```
```
1500
```
```
Graph Build Time (s)
```
```
BigANN
```
```
10 20 30 40 50 60
Combined Fanout (num clusters * fanout)
```
```
0
```
```
1000
```
```
2000
```
```
Graph Build Time (s)
```
```
Wikipedia
```
```
Replication Fanout Multi-Level Fanout
```
```
Figure 3: Multi-level fanout significantly reduces build times, with the
effect becoming more pronounced as fanout is increased.
```
A.2 for a more detailed explanation. This yields massive speedups
overall, because the large number of leaders per subproblem causes
the recursion to terminate after very low depth (often, a mere 2 to 3
steps of recursion suffices). That is, the topmost levels of recursion
constitute the majority of the partitioning time.
We ablate the construction time yielded by these three strategies.
Using fanout instead of replication yields an average 1. 47 ×speedup
in partitioning time, and multi-level fanout yields an average 3. 35 ×
speedup (see Supplemental Figure 9). Figure 3 shows how these
speedups impact total construction time. Further, we find that repli-
cation, fanout, and multi-level fanout are equivalent with respect
to the quality of the resultant graph (Supplemental Figure 8).
Leaf-Building Optimizations. A key benefit of the partition-
based approach of PiPNN is that it enables the use of optimized
matrix multiplication and sorting routines to quickly prune candi-
date lists. In particular, we compute a distance matrix containing
all distances between points in a leaf via the Eigen library [ 20 ]. We
further optimize leaf building by obtaining the nearest neighbors of
each point in a leaf using a vectorized partial sort implementation
from the Highway library [ 17 ]. In total, these optimizations com-
bine to yield a 27 ×speedup in leaf construction time. For a more
detailed treatment of these optimizations, see Supplement A.4.
Parameter Tuning HashPrune. We evaluated the effect of using
more or fewer hash functions in HashPrune. Using fewer hash
functions makes collisions more likely, thus in effect broadening the
angle of difference that two points must have relative to the source
point. On the other hand using more hash functions could quickly
make collisions very unlikely even for nearby points. We evaluated
the performance of PiPNN indexes built with𝑏 ∈ { 6 , 8 , 10 , 12 , 14 , 16 }
hash functions. We saw that a broad range of values are acceptable,
with only the use of 6 hash functions degrading performance, as
displayed in Supplemental Figure 13. We chose to use 12 as the
default for our experiments.
Running Time Breakdown. For reference, we provide a break-
down of the time spent in various phases of PiPNN, namely par-
titioning into leaves (Partition), building𝑘-NN graphs over those
leaves and passing the edges into HashPrune (Build Leaves), and

```
0% 20% 40% 60% 80% 100%
% of total build time
```
```
SpaceV
OpenAI
BigANN
Wiki
```
```
57% 35% 8%
29% 54% 17%
56% 35% 8%
67% 22% 11%
```
```
Partition Build Leaves Final Prune
```
```
Figure 4: Portion of time spent in Partitioning, Leaf Building, and
Final Prune for the four ablation datasets.
```
```
performing a prune on the final HashPrune reservoirs (Final Prune).
We show the percentage of time taken by each phase in Figure 4.
```
### 5.2 PiPNN vs. Existing ANNS Methods

```
We now turn to evaluating the best instance of PiPNN that we
identified against state-of-the-art methods Vamana and HNSW, as
well as HCNNG. We demonstrate that PiPNN consistently produces
indexes of comparable quality to the best across all datasets. We then
show that PiPNN always produces indexes much more quickly than
rival methods, and document that as expected the superior data-
locality of PiPNN contributes to this result. We conclude by applying
the considered ANNS methods to the downstream task of building
a high quality𝑘−NN graph, where PiPNN’s fast construction times
lead to a more than 2x speedup over HNSW.
Index Quality. The results of our experiments are shown in Fig-
ure 5. The runs on OpenAI and WikiCohere show that PiPNN with a
single replica yields graph indexes of equivalent quality to Vamana
with one pass. The improved performance that Vamana achieves by
adding an extra pass can be matched in PiPNN by performing an
extra replicate, i.e., an independent copy of the RBC algorithm. This
behavior remains consistent on all of our billion-scale datasets.
Our runs with an extra replicate show that PiPNN is able to trade
additional computation to build higher quality indexes with corre-
sponding increases in the build time. We note that such a tradeoff
is not possible for HCNNG. In particular, as HCNNG increases the
number of replicas, it unavoidably increases the average degree (and
maximum degree) of the graph, increasing the number of distance
comparisons and degrading query performance. Additional replicas
in HCNNG also require buffering three candidates per point, per
replica, and thus quickly exceed reasonable memory limits.
MIRAGE was unable to run on billion-scale datasets due to run-
ning out of memory. MIRAGE was able to run on our 100M-scale
datasets, but was unable to achieve above 0.8 recall. We note that
in the MIRAGE paper, the authors only evaluated their algorithm
on very small (i.e. million-scale) datasets. The authors of FastKCNA
also evaluated their work on million-scale datasets, though we were
able to run their work on up to 100M points without running out
of memory. We found FastKCNA achieved on average 0 .78% query
throughput across BigANN-100M and MS-SPACEV-100M as com-
pared with PiPNN and took average 18. 64 ×longer to build. Our
results show that PiPNN is significantly faster than all other meth-
ods. The highest quality indexes constructed by PiPNN (2-replica
```

```
Tobias Rubel, Richard Wen, Laxman Dhulipala, Lars Gottesbüren, Rajesh Jayaram, and Jakub Łącki
```
```
Figure 5: QPS vs recall for all algorithms on four billion-size datasets (BigANN-1B, MS-SPACEV-1B, MS-TURING-1B, and DEEP-1B) and two
high-dimensional datasets (OpenAIArXiv-2M and WikipediaCohere-35M). Build times are listed in the legends in seconds.
```
PiPNN) are always built faster than 1-pass Vamana, while always
being superior in quality (typically equivalent to 2-pass Vamana).
Index Construction Times. On the very high dimensional datasets
OpenAI and WikiCohere, we observe that PiPNN consistently yields
significant speedups over HCNNG, HNSW, Vamana, MIRAGE, and
FastKCNA. PiPNN builds indexes 8-20×faster than other methods,
or 4-10×faster when performing an extra replica for better query
performance. On the billion-scale datasets, PiPNN builds 4-12×
faster than HCNNG and Vamana with a single replica, and 2-6×
faster when doing an extra replica.
Memory Accesses and Instructions. PiPNN’s performant index
construction arises from better data locality leading to an order of
magnitude fewer cycles and LLC-misses than Vamana and HNSW.
It thus achieves superior instructions per cycle (1.26 vs. 0.44 and
0.33). Additionally, the use of SIMD yields an improvement in the
number of instructions needed. We give a breakdown of the cycles,
instructions, and last level cache misses for PiPNN, Vamana, and
HNSW during index construction in Supplemental Table 4.
Building𝑘-NN Graphs. As an application of PiPNN’s faster
index construction, we consider the problem of constructing a𝑘-
NN graph for𝑘= 10 that has at least 95% recall for the𝑘-NN edges.
We note that building a high-quality𝑘-NN graph is a key substep
of numerous unsupervised clustering algorithms [ 13 , 14 , 35 , 42 ],
as well as large-scale data de-duplication pipelines [ 11 , 19 ]. We
performed a grid search over parameters used to tune index quality
and search quality for each algorithm to minimize the total time to
construct the final 𝑘 -NN graph.
Figure 6 shows the results of our𝑘-NN construction, showing the
overall slowdown of each method over PiPNN, which is always the

```
BigANN-100MSPACEV-100MDeep-100MTuring-100M
```
```
OpenAI
```
```
0
```
```
2
```
```
4
```
```
6
```
```
Slowdown (relative to fastest)
```
```
k -NN Building Algorithm
hnswlib Vamana PiPNN
```
```
Figure 6: Performance of𝑘-NN graph building for𝑘= 10 using
different ANNS methods. The𝑘-NN graphs have at least 95% recall.
```
```
fastest. Although PiPNN yields much larger speedups if one simply
measures the index build, in this application, this task requires
not only building an index, but to query it for all points, and the
time for this latter step is roughly the same for all high-quality
indexing methods. We find that compared to hnswlib’s HNSW
implementation PiPNN is between 2. 2 – 6. 9 x faster. Compared to
ParlayANN’s highly optimized Vamana implementation, PiPNN is
between 1. 4 – 1. 7 x faster in its end-to-end time.
```
### 6 DISCUSSION AND CONCLUSION

```
With PiPNN, we have devised a novel partitioning-based algo-
rithm for constructing graph-based nearest neighbor search indexes
which builds indexes up to 12.9x faster than the prior state of the
art. This enables significant speedups for downstream tasks, as
shown in our application experiment, wherein we achieve up to
```

```
PiPNN : A Framework for Ultra-Scalable Graph-Based Nearest Neighbor Indexing
```
6.9x speedups over HNSW for high quality𝑘−NN construction.
Moreover, we have shown in our ablation study that the general
framework of PiPNN admits a rich design space, and multiple strate-
gies can be employed to design a family of competitive algorithms.
Our HashPrune algorithm makes online pruning extremely easy
and, by guaranteeing a fixed degree, allows us to trade more compu-
tation for better index quality which results in index quality on-par
with 2-pass Vamana.
Our results open several promising lines for future study. Firstly,
it remains to be shown what further benefits we could get by use of
quantized GEMM operations (e.g., on scalar-quantized points). Sec-
ondly, since the basic operation of PiPNN is matrix multiplication,
we could make use of hardware acceleration from GPUs or TPUs to
further accelerate index construction. Moreover, while HashPrune
alone yields high quality indexes, it remains to be seen whether the
method can be augmented to achieve its best performance without
the use of a final prune. Lastly, we believe that PiPNN’s approach
is a natural fit for distributed data processing, and are interested in
studying how PiPNN can scale when building graphs on very large
scale inputs (e.g., tens of billions of points).

### 7 ACKNOWLEDGEMENT

```
This work is supported by NSF grants CCF-2403235 and CNS
and by the National Science Foundation Graduate Research Fel-
lowship Program under Grant No. DGE 2236417. Any opinions,
findings, and conclusions or recommendations expressed in this
material are those of the author(s) and do not necessarily reflect
the views of the National Science Foundation.
```
### REFERENCES

```
[1]Alexandr Andoni, Piotr Indyk, Thijs Laarhoven, Ilya P. Razenshteyn, and Ludwig
Schmidt. 2015. Practical and Optimal LSH for Angular Distance. 1225–1233.
[2]Alexandr Andoni and Hengjie Zhang. 2023. Sub-quadratic (1+𝜖)-approximate
Euclidean Spanners, with Applications. In 2023 IEEE 64th Annual Symposium on
Foundations of Computer Science (FOCS). IEEE, 98–112.
[3]Sunil Arya and David M Mount. 1993. Approximate nearest neighbor queries in
fixed dimensions.. In SODA, Vol. 93. Citeseer, 271–280.
[4]Martin Aumüller, Erik Bernhardsson, and Alexander Faithfull. 2020. ANN-
Benchmarks: A benchmarking tool for approximate nearest neighbor algorithms.
Information Systems 87 (2020), 101374.
[5]Amir Azarmehr, Soheil Behnezhad, Rajesh Jayaram, Jakub Łącki, Vahab Mirrokni,
and Peilin Zhong. 2025. Massively parallel minimum spanning tree in general
metric spaces. In Proceedings of the 2025 Annual ACM-SIAM Symposium on
Discrete Algorithms (SODA). SIAM, 143–174.
[6]Artem Babenko and Victor Lempitsky. 2016. Efficient indexing of billion-scale
datasets of deep descriptors. In Proceedings of the IEEE conference on computer
vision and pattern recognition. 2055–2063.
[7]Yair Bartal, Moses Charikar, and Danny Raz. 2001. Approximating min-sum
k-clustering in metric spaces. In Proceedings of the thirty-third annual ACM
symposium on Theory of computing. 11–20.
[8]MohammadHossein Bateni, Soheil Behnezhad, Mahsa Derakhshan, Moham-
madTaghi Hajiaghayi, Raimondas Kiveris, Silvio Lattanzi, and Vahab Mirrokni.
```
2017. Affinity clustering: Hierarchical clustering at scale. Advances in Neural
Information Processing Systems 30 (2017).
[9]Lorenzo Beretta, Vincent Cohen-Addad, Rajesh Jayaram, and Erik Waingarten.
2025. Approximating High-Dimensional Earth Mover’s Distance as Fast as
Closest Pair. arXiv preprint arXiv:2508.06774 (2025).
[10]Alina Beygelzimer, Sham M. Kakade, and John Langford. 2006. Cover trees
for nearest neighbor (ACM International Conference Proceeding Series), Vol. 148.
ACM, 97–104.
[11]CJ Carey, Jonathan Halcrow, Rajesh Jayaram, Vahab Mirrokni, Warren Schudy,
and Peilin Zhong. 2022. Stars: Tera-Scale Graph Building for Clustering and
Learning. Advances in Neural Information Processing Systems 35 (2022), 21470–
21481.
[12]Moses Charikar. 2002. Similarity estimation techniques from rounding algo-
rithms. In Proceedings on 34th Annual ACM Symposium on Theory of Computing,

```
May 19-21, 2002, Montréal, Québec, Canada. ACM, 380–388.
[13]Laxman Dhulipala, David Eisenstat, Jakub Łącki, Vahab Mirrokni, and Jessica
Shi. 2021. Hierarchical agglomerative graph clustering in nearly-linear time. In
International conference on machine learning. PMLR, 2676–2686.
[14]Laxman Dhulipala, Jakub Łącki, Jason Lee, and Vahab Mirrokni. 2023. Terahac:
Hierarchical agglomerative clustering of trillion-edge graphs. Proceedings of the
ACM on Management of Data 1, 3 (2023), 1–27.
[15]Cong Fu, Chao Xiang, Changxu Wang, and Deng Cai. 2019. Fast Approximate
Nearest Neighbor Search With The Navigating Spreading-out Graph. Proceedings
of the VLDB Endowment 12, 5 (2019), 461–474. https://doi.org/10.14778/3303753.
3303754
[16]Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yixin
Dai, Jiawei Sun, Haofen Wang, and Haofen Wang. 2023. Retrieval-augmented
generation for large language models: A survey. arXiv preprint arXiv:2312.
2, 1 (2023).
[17]Google. 2026. Highway: Performance-portable SIMD. https://github.com/
google/highway
[18]Lars Gottesbüren, Laxman Dhulipala, Rajesh Jayaram, and Jakub Lacki. 2025.
Unleashing Graph Partitioning for Large-Scale Nearest Neighbor Search. Proc.
VLDB Endow. 18, 6 (2025), 1649–1662.
[19]Yang Gu, Caroline Zhou, Qiao Zhang, Scott Wang, Yongzhe Wang, Li Zhang,
Nikos Parotsidis, Cj Carey, Ashkan Fard, Mingyan Gao, et al.2025. Streaming
Trends: A Low-Latency Platform for Dynamic Video Grouping and Trending Cor-
pora Building. In Proceedings of the Nineteenth ACM Conference on Recommender
Systems. 1091–1094.
[20]Gaël Guennebaud, Benoît Jacob, et al.2010. Eigen v3. http://eigen.tuxfamily.org.
[21]Piotr Indyk and Rajeev Motwani. 1998. Approximate nearest neighbors: towards
removing the curse of dimensionality. In Proceedings of the thirtieth annual ACM
symposium on Theory of computing. 604–613.
[22]Suhas Jayaram Subramanya, Fnu Devvrit, Harsha Vardhan Simhadri, Ravishankar
Krishnawamy, and Rohan Kadekodi. 2019. Diskann: Fast accurate billion-point
nearest neighbor search on a single node. Advances in neural information pro-
cessing Systems 32 (2019).
[23]Hervé Jégou, Matthijs Douze, and Cordelia Schmid. 2011. Product quantization
for nearest neighbor search. IEEE transactions on pattern analysis and machine
intelligence 33, 1 (2011), 117–128.
[24]Ravishankar Krishnaswamy, Magdalen Dobson Manohar, and Harsha Vardhan
Simhadri. 2024. The DiskANN library: Graph-Based Indices for Fast, Fresh and
Filtered Vector Search. IEEE Data Eng. Bull. (2024).
[25]Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir
Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, et al.2020. Retrieval-augmented generation for knowledge-intensive nlp
tasks. Advances in Neural Information Processing Systems 33 (2020), 9459–9474.
[26]Yuliang Li, Jinfeng Li, Yoshihiko Suhara, AnHai Doan, and Wang-Chiew Tan.
```
2020. Deep entity matching with pre-trained language models. Proc. VLDB
Endow. 14, 1 (2020), 50–60.
[27]Yu A Malkov and Dmitry A Yashunin. 2018. Efficient and robust approximate
nearest neighbor search using hierarchical navigable small world graphs. IEEE
transactions on pattern analysis and machine intelligence 42, 4 (2018), 824–836.
[28]Magdalen Dobson Manohar, Zheqi Shen, Guy Blelloch, Laxman Dhulipala, Yan
Gu, Harsha Vardhan Simhadri, and Yihan Sun. 2024. Parlayann: Scalable and
deterministic parallel graph-based approximate nearest neighbor search algo-
rithms. In Proceedings of the 29th ACM SIGPLAN Annual Symposium on Principles
and Practice of Parallel Programming. 270–285.
[29]Bhaskar Mitra, Nick Craswell, et al.2018. An introduction to neural information
retrieval. Foundations and Trends® in Information Retrieval 13, 1 (2018), 1–126.
[30]Javier Vargas Munoz, Marcos A Gonçalves, Zanoni Dias, and Ricardo da S Torres.
2019. Hierarchical clustering-based graphs for large scale approximate nearest
neighbor search. Pattern Recognition 96 (2019), 106970.
[31]Arvind Neelakantan, Tao Xu, Raul Puri, Alec Radford, Jesse Michael Han, Jerry
Tworek, Qiming Yuan, Nikolas Tezak, Jong Wook Kim, Chris Hallacy, et al.
2022. Text and code embeddings by contrastive pre-training. arXiv preprint
arXiv:2201.10005 (2022).
[32]Ninh Pham and Tao Liu. 2022. Falconn++: A Locality-sensitive Filtering Approach
for Approximate Nearest Neighbor Search. CoRR abs/2206.01382 (2022). https:
//doi.org/10.48550/arXiv.2206.01382 arXiv:2206.
[33]Nils Reimers. 2022. Cohere/wikipedia-22-12-en-embeddings. https://huggingface.
co/datasets/Cohere/wikipedia-22-12-en-embeddings.
[34]Deepjyoti Roy and Mala Dutta. 2022. A systematic review and research perspec-
tive on recommender systems. Journal of Big Data 9, 1 (2022), 59.
[35]Jessica Shi, Laxman Dhulipala, David Eisenstat, Jakub Łăcki, and Vahab Mir-
rokni. 2021. Scalable community detection via parallel correlation clustering.
Proceedings of the VLDB Endowment 14, 11 (2021), 2305–2313.
[36]Harsha Vardhan Simhadri, Martin Aumüller, Amir Ingber, Matthijs Douze,
George Williams, Magdalen Dobson Manohar, Dmitry Baranchuk, Edo Liberty,
Frank Liu, Ben Landrum, et al.2024. Results of the Big ANN: NeurIPS’23 compe-
tition. arXiv preprint arXiv:2409.17424 (2024).


```
Tobias Rubel, Richard Wen, Laxman Dhulipala, Lars Gottesbüren, Rajesh Jayaram, and Jakub Łącki
```
[37]Harsha Vardhan Simhadri, George Williams, Martin Aumüller, Matthijs Douze,
Artem Babenko, Dmitry Baranchuk, Qi Chen, Lucas Hosseini, Ravishankar Krish-
naswamny, Gopal Srinivasa, et al.2022. Results of the NeurIPS’21 challenge on
billion-scale approximate nearest neighbor search. In NeurIPS 2021 Competitions
and Demonstrations Track. PMLR, 177–189.
[38]Jack Spalding-Jamieson, Eliot Wong Robson, and Da Wei Zheng. 2025. Scalable k-
Means Clustering for Large k via Seeded Approximate Nearest-Neighbor Search.
arXiv preprint arXiv:2502.06163 (2025).
[39]Sairaj Voruganti and M. Tamer Özsu. 2025. MIRAGE-ANNS: Mixed Approach
Graph-based Indexing for Approximate Nearest Neighbor Search. Proc. ACM
Manag. Data 3, 3, Article 188 (June 2025), 27 pages. https://doi.org/10.1145/
3725325
[40]Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul N. Bennett,
Junaid Ahmed, and Arnold Overwijk. 2021. Approximate Nearest Neighbor Neg-
ative Contrastive Learning for Dense Text Retrieval. In International Conference
on Learning Representations, ICLR.
[41]Shuo Yang, Jiadong Xie, Yingfan Liu, Jeffrey Xu Yu, Xiyue Gao, Qianru Wang,
Yanguo Peng, and Jiangtao Cui. 2024. Revisiting the index construction of
proximity graph-based approximate nearest neighbor search. arXiv preprint
arXiv:2410.01231 (2024).
[42]Shangdi Yu, Laxman Dhulipala, Jakub Łącki, and Nikos Parotsidis. 2025. Dyn-
HAC: Fully Dynamic Approximate Hierarchical Agglomerative Clustering. In
Proceedings of the 2025 SIAM International Conference on Data Mining (SDM).
SIAM, 252–260.
[43]Shangdi Yu, Joshua Engels, Yihao Huang, and Julian Shun. 2025. Pecann: Paral-
lel efficient clustering with graph-based approximate nearest neighbor search.
In 2025 Proceedings of the Conference on Applied and Computational Discrete
Algorithms (ACDA). SIAM, 1–17.
[44]Alexandros Zeakis, George Papadakis, Dimitrios Skoutas, and Manolis
Koubarakis. 2023. Pre-Trained Embeddings for Entity Resolution: An Experi-
mental Analysis. Proc. VLDB Endow. 16, 9 (2023), 2225–2238.
[45]Xi Zhao, Yao Tian, Kai Huang, Bolong Zheng, and Xiaofang Zhou. 2023. Towards
efficient index construction and approximate nearest neighbor search in high-
dimensional spaces. Proceedings of the VLDB Endowment 16, 8 (2023), 1979–1991.


```
PiPNN : A Framework for Ultra-Scalable Graph-Based Nearest Neighbor Indexing
```
### A APPENDIX

### A.1 Partitioning Methods

```
In this section, we study several different methods for generating
overlapping partitions of the point-sets including: (1) random binary
partitioning; (2) the RBC method described in the main text; (3)
hierarchical k-means; and (4) sorting-LSH. We then experimentally
evaluate our choices for partitioning across different datasets and
show that RBC is a high-quality and scalable default partitioning
method for PiPNN.
```
```
A.1.1 Binary Partitioning. Given a subproblemP ⊆ X, binary
partitioning selects two distinct leader points𝑝 1 ,𝑝 2 ∈ Pat random,
then dividingPinto two partsP 1 = {𝑝 ∈ P:∥𝑝,𝑝 1 ∥ ≤ ∥𝑝,𝑝 2 ∥}
andP 2 = {𝑝 ∈ P:∥𝑝,𝑝 1 ∥> ∥𝑝,𝑝 2 ∥}[ 30 ]. In plain words, we
assign each point to its closest leader. The two resulting sets de-
fine the points in the subsequent subproblems, which are divided
recursively by applying binary partitioning. This repeats until the
number of points in every subproblem is at most a predefined clus-
ter size𝐶. The resulting clusters,B 1 ,.. .,B𝑏, we refer to as leaves.
Binary partitioning is the only technique used in the HCNNG
algorithm; after dividing all points into leaf clusters, HCNNG pro-
ceeds by building subgraphs on every leaf, then taking the union
of these subgraphs. Because the leaves resulting from a single par-
titioning procedure are disjoint, the subgraphs built on them will
naturally be disconnected from each other. Thus, the algorithm
needs some way to connect leaf clusters together. HCNNG resolves
this through the replication process described earlier in this paper.
That is, it runs the partitioning and leaf-building phases many times,
using different seeds for randomization in order to produce differ-
ent leaf clusters, then takes the union of all resulting graphs. This
connects the graph because, across many replications, an individual
point will share leaves with many different sets points.
While replication can be effective for combining leaves into a
unified graph, it is expensive. HCNNG often requires upwards of
30 replications to produce graphs of acceptable quality. Notably,
binary partitioning does not accommodate the fanout strategy in
lieu of replication.
```
A.1.2 Hierarchical𝑘-Means. In the well-studied problem of𝑘-
means clustering, the problem is to select a subset of𝑘centers
fromR𝑑that minimizes the sum of squared distances between any
point and its nearest center. Although𝑘-means clustering is often
difficult in high-dimensional spaces, there are many heuristic so-
lutions that work well in practice. We implement a partitioning
technique based on hierarchical𝑘-means by applying Alg. 5, but
rather than selecting random leaders in each subproblem, we in-
stead run a𝑘-Means algorithm to select𝑘cost-minimizing leaders
which we then use for ball-carving. The remainder of the algo-
rithm is the same, including the fanout procedure. We find that
choosing leaders through𝑘-means yields a graph of similar (but
slightly worse) quality to choosing leaders at random (see Figure 7
in Section 5).

```
A.1.3 Sorting LSH. Locality-Sensitive Hashing (LSH) is a cele-
brated technique for clustering points for the purpose of nearest
neighbor search [ 1 , 12 , 32 ]. A LSH familyHis a family of hash
functions of the formℎ:X →{ 0 , 1 }, such that similar points are
```
```
Table 2: Time spent on each partitioning method. Listed times are
in seconds and only include the partitioning phase.
Method BigANN SPACEV OpenAI Wiki
Binary Partitioning 210.3 183.4 99.3 971.
Rand. Ball Carving 224.3 220.7 10.2 417.
𝑘 -Means Clustering 896.6 856.4 110.0 N/A
Sorting LSH 469.5 369.9 110.7 1212.
```
```
more likely to collide; namely, the probability𝑃𝑟ℎ∼H[ℎ(𝑥)=ℎ(𝑦)]
should be large when𝑥,𝑦 ∈ Xare similar, and small when𝑥,𝑦
are farther apart. Following [ 18 ], we create a Sorting LSH index
by hashing each point𝑥 ∈ 𝑅multiple times via independent hash
functionsℎ 1 , ...,ℎ𝑡fromH, concatenating the hashes into a string
(ℎ 1 (𝑥),ℎ 2 (𝑥), ...,ℎ𝑡(𝑥)), and then sorting the points in𝑅lexico-
graphically based on these strings of hashes. We then partition into
leaves by dividing the sorted sequence into consecutive groups of
at most𝑚𝑥 points.
Since there is no natural analog of fanout for a sorting LSH
index, we rely on replication instead of fanout to place a point in
multiple leaves. Although constructing a sorting LSH index is quite
fast (it requires applying a number of hyper-plane tests per-point,
and some parallel sorting), the extra cost of performing replication
when compared to multi-level fanout offsets much of the faster
computation of LSH-regions compared to dense-ball carving, to
the point of actually being slower. We further find that LSH-based
partitioning yields a significantly lower quality graph than𝑘-ball
carving, either with random leaders or𝑘-means leaders (Figure 7).
Comparison of partitioning strategies. We have now described
four partitioning methods for use in PiPNN: binary partitioning,
dense-ball carving with either random leaders or𝑘-means centroids,
and Sorting-LSH. To demonstrate the merits of the partitioning
methods in isolation, we fix the leaf strategy to the best perform-
ing method (bidirected𝑘-NN). The relative performance between
partitioning strategies was similar when using other leaf methods.
Fig. 7 shows the QPS-recall tradeoff of the different partitioning
methods on the ablation datasets. Randomized ball carving consis-
tently produces the best query performance. Binary partitioning
and𝑘-means are sometimes competitive (OpenAI, BigANN). Table
2 shows the time taken by the different partitioning methods. Ran-
domized ball carving is consistently much faster than𝑘-means and
Sorting-LSH. Compared to binary partitioning it is substantially
faster on OpenAI and Wiki, while exhibiting similar running time
on BigANN and SpaceV. We omit the results for𝑘-means clustering
on Wikipedia Cohere because it failed to complete within 24 hours.
Although𝑘-means could be accelerated, e.g., with subsampling
strategies, as the index quality is not better than ball carving we did
not optimize the indexing time further. In summary, randomized
ball carving consistently achieves the highest quality indexes while
having the fastest construction time.
```
```
A.1.4 Fanout vs. Replicas. In Sec. 4.1, we explain that PiPNN’s RBC
method uses a single recursive process to select multiple leaders
per level, referred to as multi-level fanout, instead of repeatedly
replicating the entire partitioning. We showed that this significantly
speeds up partitioning times Fig. 3 and had no impact on quality.
We show the lack of quality impact in Fig. 8. As can be seen, the
difference in QPS at 0. 9 recall is marginal and without any consistent
```

```
Tobias Rubel, Richard Wen, Laxman Dhulipala, Lars Gottesbüren, Rajesh Jayaram, and Jakub Łącki
```
```
Figure 7: Four different partitioning methods applied in the PiPNN
algorithm to different datasets. All four methods use a 2NN-graph
in the leaf-building phase.𝑘-means is omitted from the WikiCohere
plot due to failure to finish within 24h.
```
winner. Any differences in the results can likely be explained by
OS jitter.

### A.2 Fanout vs. Replication

In Section 4.1, we presented fanout and multi-level fanout as replace-
ments to replication for strategies to create overlapping partitions.
Here, we give a more thorough explanation for why fanout and
multi-level fanout yield significantly better build times.
Suppose we selectℓleaders in each subproblem during parti-
tioning. Then within a given level of the recursion tree, we must
obtain the distances between each point and the leaders within the
subproblems in which it is contained. In other words, we compute a
total of𝑓𝑖·|P|·ℓdistances in the𝑖th level, where𝑓𝑖is the aggregate
fanout (i.e. how many subproblems each point is contained in) at
level𝑖. Under the replication strategy,𝑓𝑖= 1 at every level, but we
repeat the entire procedure some𝑟many times, yielding the same
cost as if each point had aggregate fanout𝑟at every level. If we
were to get an analogous partitioning under the fanout strategy, we
would execute the top level only once, wherein we would assign
each point to the subproblems of its𝑟nearest leaders. Levels be-
yond the first would have𝑓𝑖= 𝑟and thus have the same number of
distance computations as with replication, but the first level wpuld
only call for|P|· ℓ. Multi-level fanout extends the cost reduction
beyond the first level. For example, if we fanout𝑟 1 on the top level
and𝑟 2 on the second level, then the second level now maintains a
low aggregate fanout of 𝑟 1.
In summary, the strategies require the following number of dis-
tance computations:
𝑟 replications
𝑟 ·|P|· ℓ each level
𝑟 fanout
|P|· ℓ first level, 𝑟 ·|P|· ℓ thereafter
𝑟 1 ,𝑟 2 multi-level fanout (where 𝑟 1 · 𝑟 2 = 𝑟 )
|P|·ℓfirst level,𝑟 1 ·|P|·ℓsecond level,𝑟·|P|·ℓthereafter
As a reminder, the recursion depth is usually very low because

```
10 20 30 40 50 60
Combined Fanout (num clusters * fanout)
```
```
0
```
```
200000
```
```
400000
```
```
600000
```
```
QPS @ Recall
```
```
≥
```
```
0.
```
```
SpaceV
```
```
10 20 30 40 50 60
Combined Fanout (num clusters * fanout)
```
```
0
```
```
5000
```
```
10000
```
```
15000
```
```
QPS @ Recall
```
```
≥
```
```
0.
```
```
OpenAI
```
```
10 20 30 40 50 60
Combined Fanout (num clusters * fanout)
```
```
0
```
```
200000
```
```
400000
```
```
QPS @ Recall
```
```
≥
```
```
0.
```
```
BigANN
```
```
10 20 30 40 50 60
Combined Fanout (num clusters * fanout)
```
```
0
```
```
20000
```
```
40000
```
```
QPS @ Recall
```
```
≥
```
```
0.
```
```
Wikipedia
```
```
Replication Fanout Multi-Level Fanout
```
```
Figure 8: Given equal point repeat budget, all three methods exhibit
similar QPS at fixed recall.
```
```
10 20 30 40 50 60
Combined Fanout (num clusters * fanout)
```
```
0
```
```
1000
```
```
2000
```
```
Aggregate Bucket Time (s)
```
```
SpaceV
```
```
10 20 30 40 50 60
Combined Fanout (num clusters * fanout)
```
```
0
```
```
50
```
```
100
```
```
Aggregate Bucket Time (s)
```
```
OpenAI
```
```
10 20 30 40 50 60
Combined Fanout (num clusters * fanout)
```
```
0
```
```
1000
```
```
2000
```
```
Aggregate Bucket Time (s)
```
```
BigANN
```
```
10 20 30 40 50 60
Combined Fanout (num clusters * fanout)
```
```
0
```
```
500
```
```
1000
```
```
1500
```
```
Aggregate Bucket Time (s)
```
```
Wikipedia
```
```
Replication Fanout Multi-Level Fanout
```
```
Figure 9: Given equal point repeat budget, Multi-level Fanout
greatly reduces time to partition into leaves (termed ‘Buckets’ here).
```
```
of the high number of leaders selected in each subproblem. Thus,
reducing the cost of the top 2 levels of recursion has a very large
impact on overall partitioning time. This also limits how many
levels we can distribute fanout across, as any fanout beyond the
second level has a high likelihood to not take effect due to many
subproblems already reaching their base cases.
```
### A.3 Picking within Leaves

```
As mentioned, PiPNN selects for each element of a leaf a subset of
its co-inhabitants to prune via HashPrune. In our final algorithm
we used bi-directed k-NN’s for this purpose. In this section we
describe all the other methods we considered, and give an ablation
to demonstrate why k-NN was suitable for inclusion as the sole
method used in PiPNN.
A.3.1 Degree-Restricted Minimum Spanning Trees. We first de-
scribe an existing leaf method developed for use in the HCNNG
algorithm that we incorporate into PiPNN. The method is to build
an undirected degree-restricted minimum spanning tree over the
```

```
PiPNN : A Framework for Ultra-Scalable Graph-Based Nearest Neighbor Indexing
```
```
points within a leaf, where the degree of each node is capped to
be at most 3. We will refer to this approach simply as MST. This
approach enforces connectivity on the subgraph while (1) priori-
tizing the shortest edges and (2) ensuring that the resulting graph
is sparse. Intuitively, these short edges are important for the final
stages of a greedy search, as they increase the likelihood of finding
the true nearest neighbor once the search has navigated to the
correct region of the graph.
However, building an MST using all pairwise distances within
a leaf is a performance bottleneck due to the cost of storing and
sorting all pairwise edges (required e.g., when running a degree-
restricted variant of Kruskal’s algorithm). To solve this, the imple-
mentation of HCNNG in ParlayANN builds the MST from a sparser
graph where each point is only connected to its𝑙-nearest neigh-
bors (e.g.,𝑙= 10 ) within the leaf. This restriction was shown not
to affect the quality of the resulting index, and yields significant
performance improvements.
```
A.3.2 𝑘-Nearest Neighbor Graphs. Extrapolating from the idea
that short-range edges are important for traversing small local
neighborhoods, a very natural subgraph to build on a leaf is a
𝑘-nearest neighbor (𝑘-NN) graph. That is, for each point𝑣, we
identify the𝑘nearest points within the cluster and add edges to
and/or from each of those points. The resulting𝑘-NN graph is
closely related to an MST but with a few distinctions. Most notably,
in contrast to the MST, the𝑘-NN graph on a leaf is not necessarily
connected. However, it’s not clear that this property is necessary,
given that leaves themselves are not connected to each other until
replication/fanout adds overlapping subgraphs. In exchange, the
𝑘 -NN graph is faster to compute and more tunable.
k-NN Leaf Design Space. We consider several choices for directing
the edges in the𝑘-NN graph. In the standard notion of a𝑘-NN graph,
which we will refer to as the Directed k-NN, we add an edge from
each point to its𝑘nearest neighbors. Alternatively, we can flip
the directions of all edges in the Inverted k-NN, in which we
add an edge to each point from its𝑘nearest neighbors. Although
unconventional, this inverted𝑘-NN construction may in fact be
better suited to building a navigable graph than the standard𝑘-
NN graph. This is because if a greedy traversal for a point in the
dataset lands on its nearest neighbor, the only node to which it can
move that makes progress to the target point is that point itself.
More generally, we can describe this relationship as the following:
if a traversal aims to reach a node, the nodes that the traversal is
most likely to try to reach it from is its nearest neighbors. Finally,
we consider the Bi-directed k-NN, the union of the directed and
inverted𝑘-NNs. In this graph, for each point, we add edges to
and from its𝑘nearest neighbors. We will also refer to k-NN as
shorthand for this method, as we find it to be the best of the three.
Our experiments (Fig. 10) show that the inverted𝑘-NN strategy
yields graphs of much higher quality than the directed𝑘-NN, but
that the bi-directed𝑘-NN is better than both. This result suggests
that the forward edges and backward edges each contribute to
the traversability of the graph. Moreover, we find that building
bi-directed𝑘-NN on leaves with the right choice of𝑘yields better
search performance than building MSTs. Through ablations, we
reveal a clear relationship between the parameter𝑘and search

```
Table 3: Average degree for each method for each ablation dataset. Degree
is capped at 64 for all methods.
Method BigANN-100M OpenAI Space-100M Wiki Avg
Directed𝑘 -NN 24.19 12.34 19.66 18.80 18.
𝑘 -NN 41.04 21.51 34.47 32.02 32.
MST 29.34 16.44 25.71 24.75 24.
Robust Prune 62.54 43.07 60.88 55.31 55.
Inverted𝑘 -NN 31.45 18.22 27.45 25.55 25.
```
```
performance. Increasing𝑘, as one might expect, increases the aver-
age degree of each node in the graph. This makes visiting a node
more costly. In exchange, it decreases the average number of nodes
visited per query. As we show in Fig. 11, this results in a sweet spot
for 𝑘 ranging from 2 to 4 which produces the best graphs.
```
```
A.3.3 All-to-All Prune. As described in Section 2.1, the Robust-
Prune procedure takes a setNof candidate neighbors for a point
𝑥 ∈Xand returns a subset ofNunder some given size limit [ 22 ].
In particular, RobustPrune attempts to choose a set of neighbors
for𝑥from the set of candidates that ensures good navigability. For
the candidate neighbors of a point𝑥, Vamana uses all points visited
during a beam search for𝑥. Typically, the vast majority of points vis-
ited during beam search are in the close vicinity of the target point.
Thus, these points can be argued to roughly approximate a subset
of the local neighborhood around𝑥. Since the leaves into which
a given𝑥falls in PiPNN also approximate the local neighborhood
around𝑥, we consider using RobustPrune to build leaves. More
specifically, for a leafB𝑖, we run RobustPrune on each𝑥 ∈ B𝑖,
supplyingB𝑖as the candidate set. This yields up to𝑚neighbors
per point in the leaf.
Optimizing RobustPrune. We note that RobustPrune can be
costly to run on large lists of candidates, because each time a neigh-
bor is chosen, it considers all other candidates for removal. In PiPNN,
we provide RobustPrune with several times more candidates than
Vamana often does (our cluster size𝐶is roughly 1000). To remedy
this, we use a lazy implementation of RobustPrune. This version
does not remove any redundant candidates upon adding a neighbor.
Instead, prior to adding a candidate𝑐to the neighbor list, we check
all previously-admitted neighbors to see if any should have pruned
away𝑐, then delete𝑐if so. Compared to the standard version, our
version can potentially omit unnecessary comparisons. Specifically,
of the comparisons that standard RobustPrune does, it only per-
forms those on candidates closer to the target than the last neighbor
added to the output neighbor list.
Comparison of Prune Kernels. We evaluate five different meth-
ods for pruning the candidate lists induced by the partitioning
process: building a bi-directed minimum spanning tree, Robust-
Prune, building a𝑘-NN graph, building an inverted𝑘-NN graph,
and a bi-directed𝑘-NN graph. Figure 10 shows the performance of
these different leaf methods, keeping the rest of the pipeline fixed,
on the ablation datasets.
We observe that bi-directing the𝑘-NN graphs is critical for ob-
taining good performance, but between bidirected𝑘-NN and bidi-
rected MST, performance is similar; bidirected𝑘-NN outperforms
bidirected MST in some high recall situations due to the tunability
of its density. RobustPrune has the disadvantage of producing much
denser graphs, as can be seen in Table 3. This ends up producing
```

```
Tobias Rubel, Richard Wen, Laxman Dhulipala, Lars Gottesbüren, Rajesh Jayaram, and Jakub Łącki
```
```
0.85 0.9 0.95 1
```
```
104
```
```
105
```
```
QPS
```
```
Wikipedia Cohere
```
```
0.85 0.9 0.95 1
```
```
103
```
```
104
```
```
105
```
```
OpenAI
```
```
0.85 0.9 0.95 1
Recall
```
```
105
```
```
106
```
```
QPS
```
```
BigANN-100M
```
```
0.85 0.9 0.95 1
Recall
```
```
105
```
```
106
```
```
SPACEV-100M
```
```
Directed KNN
KNN
```
```
MST
All-to-All Prune
```
```
Inverted KNN
```
```
Figure 10: Comparing QPS-recall trade-offs of leaf pruning meth-
ods. Bi-directed 𝑘 -NN (𝑘 -NN) performs best.
```
indexes that require fewer visited nodes at high recall, but the in-
creased number of comparisons when visiting a node outweighs the
benefit. RobustPrune in the leaves produces candidate lists that are
unlikely to be further sparsified by either HashPrune or the final ap-
plication of RobustPrune. Thus RobustPrune produces graphs with
average degree close to the maximum degree (55.45 on average).
Tuning𝑘-NN. By increasing the value of the parameter𝑘, we
can raise the density of the subgraph produced in each leaf. This
increases the cost of visiting a node while decreasing the number
of steps taken during a query. Fig. 11 displays this effect on MS-
SPACEV across𝑘 ∈ [ 1 , 10 ]. The behavior is similar on other datasets.
When increasing𝑘, the average number of visited nodes per search
goes down sharply at first but with quickly diminishing returns.
The initial drop is indicative of the value of these first few short
edges to the general navigability of the graph – that is, only having
an edge to one’s nearest neighbor is insufficient. However, after
only a few increments of𝑘, the additional edges no longer clearly
improve the speed at which queries converge to the target. On the
other hand, the additional edges do continue to steadily increase
the number of distance computations due to further densifying the
graph.
In summary, the effects of increasing𝑘from 1 to 10 have a
significant impact on overall query performance. Increasing𝑘from
1 to{ 2 , 3 }yields an immediate increase to query throughput, but
subsequent increases yield a steady decline in throughput due to
adding redundant edges. Based on these results, we set𝑘= 2 as a
sane default, although 3 or 4 would also suffice.

### A.4 Ablation of Leaf Optimizations

```
In Section 5.1, we described several optimizations for building leaves
in PiPNN, namely building distance matrices, computing those
```
```
Figure 11: Varying the parameter 𝑘 for 𝑘 -NN graphs in leaves.
```
```
bigann openai spacev wiki
Dataset
```
```
0
```
```
10
```
```
20
```
```
30
```
```
40
```
```
50
```
```
60
```
```
Normalized Build Time (Slowdown Factor)
```
```
Normalized Build Leaf Time Comparison
```
```
Methods
All-to-All Prune
All-to-All Prune (D)
```
```
All-to-All Prune (D,E)
All-to-All Prune (D,E,VQ)
```
```
KNN
KNN (D)
```
```
KNN (D,E)
KNN (D,E,VQ)
Figure 12: Effect of leaf optimizations on total leaf building times.
(D) denotes pre-computing the distance matrix. (E) denotes pre-
computing the distance matrix with the Eigen library. (VQ) denotes
using VQSort or VQPartialSort to sort distances. Data is normalized
with respect to the fastest build time; lower is better.
```
```
matrices using the Eigen library [ 20 ], and finding near neighbors
using the Highway library’s vectorized partial sort [17].
We show the effect of these optimizations in Figure 12 on the
ablation datasets. We compare naive implementations of𝑘-NN as
well as All-to-All Prune (wherein we perform RobustPrune within
each leaf to generate candidates for HashPrune) against methods
optimized by first building a distance matrix (D), building a distance
matrix with the Eigen library (D,E), and methods which addition-
ally use VQSort or VQPartialSort for identifying near-neighbors.
Building a distance matrix gives average speedups of 2. 44 ×and
4. 95 ×for𝑘-NN and All-to-All Prune respectively. Doing so with
Eigen gives a further 8. 57 ×and 1. 98 ×speedup for the two meth-
ods. All-to-All Prune benefits marginally from using VQSort ( 1. 05 ×
speedup), whereas𝑘-NN does not get further improvement com-
pared to the priority queue approach. All optimizations combined
give our final versions of𝑘-NN and All-to-All Prune 27. 03 ×and
12. 71 × speedups respectively on average.
```
### A.5 Intermediate and Final Pruning

```
HashPrune. We developed the HashPrune procedure to ensure
that memory usage was bounded regardless of the amount of repli-
cation or the degree in a leaf. On the one hand, this is necessary to
build high quality indexes on billion-scale datasets where memory
requirements are high, but HashPrune also ensures that memory
can be allocated up-front in a single contiguous block. Thus, the
```

```
PiPNN : A Framework for Ultra-Scalable Graph-Based Nearest Neighbor Indexing
```
```
0. 9 1. 0
Recall
```
```
105
```
```
106
QPS
```
```
QPS
```
```
0. 9 1. 0
Recall
```
```
103
```
```
104
```
```
Dist Comps
```
```
Dist Comps
```
```
0. 9 1. 0
Recall
```
```
101
```
```
102
```
```
103
```
```
Avg Visited
```
```
Average Visited
```
```
6 bits 8 bits 10 bits 12 bits 14 bits 16 bits
Figure 13: Parameter sweep over hash-family sizes in the range
of[ 6 , 16 ]on SPACEV (100M) dataset, without any final prune. All
values between 8 and 16 are suitable.
```
procedure not only enables better scalability, but it also avoids
dynamically resizing the intermediate candidate lists.
To test whether limiting the candidates during construction
impacts query performance, we compared reservoir sizes of 64,
128, and 192 against an unbounded degree intermediate graph on
the ablation datasets (with final prune performed down to 64 max
degree). We observed that HashPrune has almost no impact on the
distance computations performed (and, thus, the QPS achieved) to
achieve high recall when paired with the bi-directed𝑘-NN method
we use for candidate picking candidates within buckets. These
results are displayed for MS-SPACEV-100M in Tab. 5. This table
also shows the marginal variation across different𝑚parameters
for the hash functions. This surprising parameter insensitivity is
likely caused by our use of a final prune.
Final Prune. PiPNN employs a final RobustPrune on each can-
didate list after the partitions have been processed. We found this
to improve QPS by an average of 10 .18% and 9 .45% at 0.9 and 0.
recall respectively across the four ablation datasets. We find that
the final pruning step comprises an average of 9 .82% of the total
build time across the four ablation datasets. This number drops to
just 6 .86% for the 100M sized ablation datasets, where the other
indexing steps are more costly. Due to its quality improvements,
and given its relatively modest impact to the overall construction
time, we find that final pruning is beneficial and always enable it.

### A.6 PiPNN vs. Recent ANNS Methods

We were unable to evaluate MIRAGE [ 39 ] and FastKCNA [ 41 ] on
billion scale datasets due to excessive memory usage. We had to
modify MIRAGE to run on 100M datasets, as described in Sec. A.9,
but did not need to require FastKCNA. We show results for our
two 100M sized datasets used in our PiPNN ablation in Fig. 15.
FastKCNA achieves slightly worse query performance than PiPNN
but with builds times at least 17. 3 ×longer. MIRAGE was unable
to achieve competitive performance, and had build times at least
19. 1 × longer then PiPNN.
Fig. 14 shows results for the methods able to run the billion-
scale Text2Image dataset [ 37 ] comprised of 200-dimensional float
vectors outfitted with the MIPS dissimilarity measure. HNSW is not
displayed due to being unable to achieve 0.8 recall. PiPNN performs
better than other methods with respect to maximum achieved recall.

##### 0. 85 0. 90 0. 95 1. 00

#### Recall

##### 105

##### 106

#### QPS

#### Text2Image-1B

```
hcnng (19094 s)
pipnn-1 (1471 s)
pipnn-2 (2746 s)
```
```
vamana-1 (5420 s)
vamana-2 (10845 s)
```
```
Figure 14: PiPNN achieves the highest recall on the difficult
Text2Image dataset.
```
```
Table 4: Average hardware counters fromperfper method and dataset.
Values are shown in hundred billions(× 1011 )for cycles, instructions, and
LLC-load-misses. Pre-fetching is disabled for Vamana in this experiment.
PiPNN indexes datasets in fewer cycles and with fewer cache-misses.
Method Stat (× 1011 ) bigann openai spacev wiki
```
```
pipnn
```
```
cycles 935.71 85.76 927.19 1562.
instructions 1170.70 90.99 1159.58 2066.
LLC-load-misses 3.08 0.29 3.14 4.
```
```
vamana
```
```
cycles 5490.51 1109.12 3849.37 17180.
instructions 5270.01 217.27 3445.63 3089.
LLC-load-misses 11.83 3.05 9.84 63.
```
```
hnsw
```
```
cycles 4913.36 735.54 4381.94 8376.
instructions 1098.43 92.08 1071.55 3849.
LLC-load-misses 12.41 6.47 10.11 74.
```
```
Table Size 64 128 192 NoRes
Hash Bits
8 3376.0 3545.0 3656.0 3445.
10 3433.0 3394.0 3614.0 3445.
12 3375.0 3559.0 3566.0 3445.
14 3531.0 3423.0 3421.0 3445.
16 3368.0 3653.0 3654.0 3445.
Table 5: Average cmps for Recall > 0.95 (SPACEV-100M Dataset)
```
### A.7 Asymptotic Analysis of PiPNN

```
We now consider the expected running time to build a PiPNN index.
First, we point out that although in some adversarial input sets
certain leader choices may lead to locally-unbalanced partitioning
across subproblems, it is reasonable in practice to assume that ran-
domized ball carving divides points across subproblems roughly
evenly in expectation. Under this assumption, recursive ball carv-
ing withℓrandom leaders in each subproblem terminates within
𝑂(logℓ𝑛)levels with high probability, where𝑛is the number of
```

```
Tobias Rubel, Richard Wen, Laxman Dhulipala, Lars Gottesbüren, Rajesh Jayaram, and Jakub Łącki
```
##### 10

```
4
```
##### 10

```
5
```
##### 10

```
6
```
```
QPS (1/s)
```
### BigANN 100M

```
FastKCNA (6091s)
MIRAGE (6645s)
PiPNN 1-Replica (348s)
```
##### 0.5 0.6 0.7 0.8 0.9 1.

```
Recall
```
##### 10

```
4
```
##### 10

```
5
```
##### 10

```
6
```
```
QPS (1/s)
```
### MS-SPACEV 100M

```
FastKCNA (6022s)
MIRAGE (6898s)
PiPNN 1-Replica (349s)
```
```
Figure 15: Results for the 100M sized ablation datasets MS-SPACEV-
100M and BigANN-100M comparing PiPNN against new recent
methods for scaleable index build.
```
input points. Within each level, up to𝑓𝑛instances of points each
undergo𝑑-dimensional distance computations with theℓleaders
in its subproblem (where𝑓is the amount of fanout performed
at the prior level of recursion). It follows that partitioning takes
𝑂(𝑑𝑓𝑛ℓ logℓ(𝑛)) time with high probability.
We stop partitioning when every leaf contains at most𝑐points.
We subsequently spend𝑂(|B𝑖|^2 𝑑)time for each leafB𝑖building a
|B𝑖|×|B𝑖|distance matrix and adding edges between each point
inB𝑖and the two points from which it has smallest distance. Thus,
we can charge each point inB𝑖for up to𝑂(|B𝑖|𝑑) ≤ 𝑂(𝑐𝑑)time.
In total, there are≤ 𝑓𝑛instances of points across all leaves, so
leaf-building in total takes 𝑂(𝑐𝑑𝑓𝑛) time.
HashPrune maintains a reservoir of up to𝑂(𝑚)candidate neigh-
bors for each point, where𝑚is the target maximum degree of the
final graph. Recall from the leaf-building step that each of the≤ 𝑓𝑛
instances of points adds a candidate edge to and from each of its

```
two nearest in-leaf neighbors. Thus, at most 4 𝑓𝑛candidate neigh-
bors are passed into HashPrune across all points. We hash each
one via LSH against roughly 12 random vectors using 12𝑂(𝑑)-time
distance computations. We subsequently compare its distance from
the target point with that of the furthest candidate in the reser-
voir (via brute force in𝑂(𝑚)time) and the candidate already in its
hash bucket if one exists (via binary search in𝑂(log𝑚)time). Thus,
HashPrune consumes 𝑂(𝑓𝑛𝑚) time.
Finally, having used HashPrune to limit the number of candi-
dates to𝑂(𝑚)per point, RobustPrune takes only𝑂(𝑚^2 )time for
each point. Thus, the final prune phase takes 𝑂(𝑛𝑚^2 ) time.
Combining all components of the algorithm produces a total
expected time complexity of 𝑂(𝑛(𝑑𝑓 ℓ logℓ(𝑛)+𝑐𝑑𝑓 + 𝑓𝑚+𝑚^2 )).
```
### A.8 Proof of Determinism of PiPNN

```
The determinism of PiPNN follows from two propositions. Firstly,
the partitioning produced by Algorithm 5 is deterministic. Thus
the candidates which are given to HashPrune are always the same
(though the order that they are merged into candidate lists by way
of HashPrune is non-deterministic). Secondly, HashPrune itself is
history independent, and so the order that points has no bearing on
the final adjacency lists for each point. Thus PiPNN is deterministic.
We prove the each proposition in this section.
Lemma A.1. Randomized Ball Carving is deterministic In each
subproblem the leaders are sampled randomly (Line 3), thus once
randomness is fixed the same leaders will always be sampled within
a given subproblem. The next subproblems are given by computing
an exact nearest neighbor computation on the leaders, and thus are
completely determined by the leaders and subproblem points. Merging
likewise can be done in such a way as to not depend on the scheduler.
Lemma A.2. HashPrune is History Independent
Let𝑝 ⊆ Xbe an arbitrary point, and consider𝑝’s HashPrune
reservoir𝑅𝑝= [𝑟 1 ,.. .,𝑟𝑠]with hash family𝐻={ℎ 1 ,.. .,ℎ𝑏}. Sup-
pose some fixed collection𝐶={𝑐 1 ,.. .,𝑐𝑖,.. .,𝑐𝑞}is to be inserted into
𝑅𝑝. We claim that𝑅𝑝will contain the𝑠elements of𝐶nearest to𝑝such
that they do not collide under𝐻. To see that this is true, consider that
if we removed the size restriction of the reservoir entirely, then each
position in𝑅𝑝would just contain the nearest point in each hash bucket
(lines 3-5 of HashPrune). Now, when we consider only the𝑠-sized
reservoir, the further member of𝑅𝑝is evicted in line 10 whenever the
reservoir is full. Thus, the sequence of insertions is irrelevant to the
final structure of 𝑅𝑝.
```
### A.9 Modifying MIRAGE

```
In our experiments, we compare against the MIRAGE nearest neigh-
bor index. For transparency, we point out that, at the time of writ-
ing this, the native MIRAGE codebase does not currently support
datasets much larger than the 10 million scale, because indices are
done using signed 32-bit integers. Thus, a maximum degree of 128,
for example, will lead to an integer overflow on the number of edges
when the number of vertices exceeds 33.6 million. In order to let
MIRAGE run on our datasets, we modify the code to use unsigned
64-bit integers instead, which could impact the performance of the
implementation. With that said, we see little noticeable effect on the
build and query speeds of the graphs. Moreover, this size integer is
used in the implementations of all other algorithms that we tested.
```

