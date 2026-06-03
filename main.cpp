#include <iostream>
#include <set>

class Graph
{
    //

    //hash_prune();
};

//Fanout();
//multi-level-fanout();


// ####BeamSearch del chat. Me parece muy ineficiente. Te lo dejo por si te sirve.

// #include <iostream>
// #include <vector>
// #include <unordered_set>
// #include <algorithm>
// #include <cmath>

// struct Node {
//     int id;
//     double x, y; // coordenadas de ejemplo
// };

// double distance(const Node& a, const Node& b) {
//     double dx = a.x - b.x;
//     double dy = a.y - b.y;
//     return std::sqrt(dx * dx + dy * dy);
// }

// std::vector<int> beamSearch(
//     const std::vector<Node>& nodes,
//     const std::vector<std::vector<int>>& graph,
//     int source,
//     int target,
//     int L
// ) {
//     std::vector<int> B = {source};      // beam actual
//     std::unordered_set<int> visited;    // V

//     while (true) {

//         // Buscar B \ V
//         int curr = -1;
//         double bestDist = 1e100;

//         for (int u : B) {
//             if (!visited.count(u)) {
//                 double d = distance(nodes[u], nodes[target]);
//                 if (d < bestDist) {
//                     bestDist = d;
//                     curr = u;
//                 }
//             }
//         }

//         if (curr == -1)
//             break; // B \ V = ∅

//         // B <- B ∪ Neighbors(curr)
//         for (int neigh : graph[curr]) {
//             if (std::find(B.begin(), B.end(), neigh) == B.end()) {
//                 B.push_back(neigh);
//             }
//         }

//         // V <- V ∪ {curr}
//         visited.insert(curr);

//         // Si |B| > L, quedarse con los L más cercanos a q
//         if ((int)B.size() > L) {
//             std::sort(B.begin(), B.end(),
//                 [&](int a, int b) {
//                     return distance(nodes[a], nodes[target]) <
//                            distance(nodes[b], nodes[target]);
//                 });

//             B.resize(L);
//         }
//     }

//     return B;
// }

// int main() {
//     std::vector<Node> nodes = {
//         {0, 0, 0},
//         {1, 1, 0},
//         {2, 2, 0},
//         {3, 1, 1},
//         {4, 2, 1}
//     };

//     std::vector<std::vector<int>> graph = {
//         {1, 3},     // vecinos de 0
//         {0, 2, 3},  // vecinos de 1
//         {1, 4},     // vecinos de 2
//         {0, 1, 4},  // vecinos de 3
//         {2, 3}      // vecinos de 4
//     };

//     int source = 0;
//     int target = 4;
//     int L = 3;

//     auto result = beamSearch(nodes, graph, source, target, L);

//     std::cout << "Beam final:\n";
//     for (int v : result)
//         std::cout << v << " ";

//     std::cout << std::endl;
// }