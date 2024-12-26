1.	Find Centre of star graph
#include <iostream>
#include <vector>
using namespace std;

int findCenter(vector<vector<int>>& edges) {
    return (edges[0][0] == edges[1][0] || edges[0][0] == edges[1][1]) ? edges[0][0] : edges[0][1];
}

int main() {
    vector<vector<int>> edges1 = {{1, 2}, {2, 3}, {4, 2}};
    vector<vector<int>> edges2 = {{1, 2}, {5, 1}, {1, 3}, {1, 4}};
    cout << findCenter(edges1) << endl; // Output: 2
    cout << findCenter(edges2) << endl; // Output: 1
    return 0;
}
2.	Find the town judge
#include <iostream>
#include <vector>
using namespace std;

int findJudge(int n, vector<vector<int>>& trust) {
    vector<int> trustCounts(n + 1, 0);
    for (auto& t : trust) {
        trustCounts[t[0]]--;
        trustCounts[t[1]]++;
    }
    for (int i = 1; i <= n; ++i) {
        if (trustCounts[i] == n - 1) {
            return i;
        }
    }
    return -1;
}

int main() {
    vector<vector<int>> trust1 = {{1, 2}};
    vector<vector<int>> trust2 = {{1, 3}, {2, 3}};
    vector<vector<int>> trust3 = {{1, 3}, {2, 3}, {3, 1}};
    cout << findJudge(2, trust1) << endl; // Output: 2
    cout << findJudge(3, trust2) << endl; // Output: 3
    cout << findJudge(3, trust3) << endl; // Output: -1
    return 0;
}
3.	Flood fill
#include <iostream>
#include <vector>
using namespace std;

void dfs(vector<vector<int>>& image, int x, int y, int originalColor, int newColor) {
    if (x < 0 || y < 0 || x >= image.size() || y >= image[0].size() || image[x][y] != originalColor) {
        return;
    }
    image[x][y] = newColor;
    dfs(image, x + 1, y, originalColor, newColor);
    dfs(image, x - 1, y, originalColor, newColor);
    dfs(image, x, y + 1, originalColor, newColor);
    dfs(image, x, y - 1, originalColor, newColor);
}

vector<vector<int>> floodFill(vector<vector<int>>& image, int sr, int sc, int color) {
    int originalColor = image[sr][sc];
    if (originalColor != color) {
        dfs(image, sr, sc, originalColor, color);
    }
    return image;
}

int main() {
    vector<vector<int>> image = {{1, 1, 1}, {1, 1, 0}, {1, 0, 1}};
    int sr = 1, sc = 1, color = 2;
    vector<vector<int>> result = floodFill(image, sr, sc, color);
    for (const auto& row : result) {
        for (int pixel : row) {
            cout << pixel << " ";
        }
        cout << endl;
    }
    return 0;
}
4.	01 matrix 
#include <iostream>
#include <vector>
#include <queue>
using namespace std;

vector<vector<int>> updateMatrix(vector<vector<int>>& mat) {
    int m = mat.size(), n = mat[0].size();
    vector<vector<int>> dist(m, vector<int>(n, INT_MAX));
    queue<pair<int, int>> q;

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (mat[i][j] == 0) {
                dist[i][j] = 0;
                q.push({i, j});
            }
        }
    }

    vector<int> dirs = {0, 1, 0, -1, 0};
    while (!q.empty()) {
        auto [x, y] = q.front();
        q.pop();
        for (int k = 0; k < 4; ++k) {
            int nx = x + dirs[k], ny = y + dirs[k + 1];
            if (nx >= 0 && nx < m && ny >= 0 && ny < n && dist[nx][ny] > dist[x][y] + 1) {
                dist[nx][ny] = dist[x][y] + 1;
                q.push({nx, ny});
            }
        }
    }
    return dist;
}
5.	Course Schedule II
#include <iostream>
#include <vector>
#include <queue>
using namespace std;

vector<int> findOrder(int numCourses, vector<vector<int>>& prerequisites) {
    vector<int> indegree(numCourses, 0), result;
    vector<vector<int>> graph(numCourses);

    for (auto& prereq : prerequisites) {
        graph[prereq[1]].push_back(prereq[0]);
        indegree[prereq[0]]++;
    }

    queue<int> q;
    for (int i = 0; i < numCourses; ++i) {
        if (indegree[i] == 0) q.push(i);
    }

    while (!q.empty()) {
        int course = q.front();
        q.pop();
        result.push_back(course);

        for (int next : graph[course]) {
            if (--indegree[next] == 0) q.push(next);
        }
    }

    return result.size() == numCourses ? result : vector<int>();
}
6.	Word Search
#include <iostream>
#include <vector>
using namespace std;

bool dfs(vector<vector<char>>& board, string& word, int i, int j, int index) {
    if (index == word.size()) return true;
    if (i < 0 || j < 0 || i >= board.size() || j >= board[0].size() || board[i][j] != word[index]) return false;

    char temp = board[i][j];
    board[i][j] = '#'; // Mark as visited
    bool found = dfs(board, word, i + 1, j, index + 1) ||
                 dfs(board, word, i - 1, j, index + 1) ||
                 dfs(board, word, i, j + 1, index + 1) ||
                 dfs(board, word, i, j - 1, index + 1);
    board[i][j] = temp; // Backtrack
    return found;
}

bool exist(vector<vector<char>>& board, string word) {
    for (int i = 0; i < board.size(); ++i) {
        for (int j = 0; j < board[0].size(); ++j) {
            if (dfs(board, word, i, j, 0)) return true;
        }
    }
    return false;
}
7.	Minimum Height Trees
#include <iostream>
#include <vector>
#include <queue>
using namespace std;

vector<int> findMinHeightTrees(int n, vector<vector<int>>& edges) {
    if (n == 1) return {0};

    vector<int> degree(n, 0);
    vector<vector<int>> graph(n);
    for (auto& edge : edges) {
        graph[edge[0]].push_back(edge[1]);
        graph[edge[1]].push_back(edge[0]);
        degree[edge[0]]++;
        degree[edge[1]]++;
    }

    queue<int> q;
    for (int i = 0; i < n; ++i) {
        if (degree[i] == 1) q.push(i);
    }

    while (n > 2) {
        int size = q.size();
        n -= size;
        for (int i = 0; i < size; ++i) {
            int node = q.front();
            q.pop();
            for (int neighbor : graph[node]) {
                if (--degree[neighbor] == 1) q.push(neighbor);
            }
        }
    }

    vector<int> result;
    while (!q.empty()) {
        result.push_back(q.front());
        q.pop();
    }
    return result;
}
8.	Accounts Merge
#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>

using namespace std;

class Solution {
public:
    unordered_map<string, string> parent;
    unordered_map<string, string> emailToName;

    string findParent(string email) {
        if (parent.find(email) == parent.end()) {
            parent[email] = email;
        }
        if (parent[email] != email) {
            parent[email] = findParent(parent[email]);
        }
        return parent[email];
    }

    void unionEmails(string email1, string email2) {
        string root1 = findParent(email1);
        string root2 = findParent(email2);
        if (root1 != root2) {
            parent[root1] = root2;
        }
    }

    vector<vector<string>> accountsMerge(vector<vector<string>>& accounts) {
        for (auto& account : accounts) {
            string name = account[0];
            for (int i = 1; i < account.size(); i++) {
                string email = account[i];
                if (emailToName.find(email) == emailToName.end()) {
                    emailToName[email] = name;
                }
                if (i == 1) continue;
                unionEmails(account[i], account[i-1]);
            }
        }

        unordered_map<string, vector<string>> mergedAccounts;
        for (auto& [email, name] : emailToName) {
            string root = findParent(email);
            mergedAccounts[root].push_back(email);
        }

        vector<vector<string>> result;
        for (auto& [root, emails] : mergedAccounts) {
            sort(emails.begin(), emails.end());
            vector<string> account = {emailToName[root]};
            account.insert(account.end(), emails.begin(), emails.end());
            result.push_back(account);
        }

        return result;
    }
};
9.	Rotting Oranges
#include <iostream>
#include <vector>
#include <queue>

using namespace std;

class Solution {
public:
    int orangesRotting(vector<vector<int>>& grid) {
        int m = grid.size();
        int n = grid[0].size();
        queue<pair<int, int>> q;
        int freshOranges = 0;
        int minutes = 0;
        
        // Initialize queue with rotten oranges
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (grid[i][j] == 2) {
                    q.push({i, j});
                } else if (grid[i][j] == 1) {
                    freshOranges++;
                }
            }
        }
        
        // Directions for adjacent cells (up, down, left, right)
        vector<pair<int, int>> directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        
        while (!q.empty() && freshOranges > 0) {
            int size = q.size();
            for (int i = 0; i < size; ++i) {
                auto [x, y] = q.front();
                q.pop();
                for (auto& dir : directions) {
                    int nx = x + dir.first, ny = y + dir.second;
                    if (nx >= 0 && nx < m && ny >= 0 && ny < n && grid[nx][ny] == 1) {
                        grid[nx][ny] = 2;
                        freshOranges--;
                        q.push({nx, ny});
                    }
                }
            }
            minutes++;
        }
        
        return freshOranges == 0 ? minutes : -1;
    }
};
10.	Pacific Atlantic Water flow
#include <iostream>
#include <vector>

using namespace std;

class Solution {
public:
    int m, n;

    void dfs(vector<vector<int>>& heights, vector<vector<bool>>& visited, int x, int y, int prevHeight) {
        if (x < 0 || x >= m || y < 0 || y >= n || visited[x][y] || heights[x][y] < prevHeight) {
            return;
        }
        visited[x][y] = true;
        dfs(heights, visited, x - 1, y, heights[x][y]);
        dfs(heights, visited, x + 1, y, heights[x][y]);
        dfs(heights, visited, x, y - 1, heights[x][y]);
        dfs(heights, visited, x, y + 1, heights[x][y]);
    }

    vector<vector<int>> pacificAtlantic(vector<vector<int>>& heights) {
        m = heights.size();
        n = heights[0].size();
        vector<vector<bool>> pacific(m, vector<bool>(n, false));
        vector<vector<bool>> atlantic(m, vector<bool>(n, false));
        <vector<int>> result;

        // Perform DFS from Pacific Ocean (top and left edges)
        for (int i = 0; i < m; ++i) {
            dfs(heights, pacific, i, 0, -1);  // Left border
            dfs(heights, atlantic, i, n - 1, -1);  // Right border
        }
        for (int j = 0; j < n; ++j) {
            dfs(heights, pacific, 0, j, -1);  // Top border
            dfs(heights, atlantic, m - 1, j, -1);  // Bottom border
        }

        // Find cells that can reach both oceans
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (pacific[i][j] && atlantic[i][j]) {
                    result.push_back({i, j});
                }
            }
        }

        return result;
    }
};
11.	Longest Substring
#include <iostream>
#include <unordered_set>
#include <string>
#include <algorithm>

using namespace std;

class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        unordered_set<char> chars;
        int left = 0, maxLength = 0;
        
        for (int right = 0; right < s.size(); ++right) {
            while (chars.find(s[right]) != chars.end()) {
                chars.erase(s[left]);
                left++;
            }
            chars.insert(s[right]);
            maxLength = max(maxLength, right - left + 1);
        }
        
        return maxLength;
    }
};
12.	Trapping Rain Water
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    int trap(vector<int>& height) {
        int n = height.size();
        if (n == 0) return 0;
        
        vector<int> leftMax(n), rightMax(n);
        leftMax[0] = height[0];
        rightMax[n - 1] = height[n - 1];
        
        for (int i = 1; i < n; ++i) {
            leftMax[i] = max(leftMax[i - 1], height[i]);
        }
        
        for (int i = n - 2; i >= 0; --i) {
            rightMax[i] = max(rightMax[i + 1], height[i]);
        }
        
        int waterTrapped = 0;
        for (int i = 0; i < n; ++i) {
            waterTrapped += min(leftMax[i], rightMax[i]) - height[i];
        }
        
        return waterTrapped;
    }
};
13.	Network delay time
#include <vector>
#include <queue>
#include <climits>

using namespace std;

class Solution {
public:
    int networkDelayTime(vector<vector<int>>& times, int n, int k) {
        vector<vector<pair<int, int>>> adj(n + 1); // adjacency list

        // Fill the adjacency list with edges
        for (auto& time : times) {
            adj[time[0]].push_back({time[1], time[2]});
        }

        // Min-heap to store the node and its time
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
        vector<int> dist(n + 1, INT_MAX);
        
        pq.push({0, k}); // Start with node k, time = 0
        dist[k] = 0;
        
        while (!pq.empty()) {
            auto [time, node] = pq.top();
            pq.pop();
            
            for (auto& neighbor : adj[node]) {
                int nextNode = neighbor.first;
                int travelTime = neighbor.second;
                
                if (time + travelTime < dist[nextNode]) {
                    dist[nextNode] = time + travelTime;
                    pq.push({dist[nextNode], nextNode});
                }
            }
        }

        int maxTime = 0;
        for (int i = 1; i <= n; ++i) {
            if (dist[i] == INT_MAX) {
                return -1; // If any node is unreachable
            }
            maxTime = max(maxTime, dist[i]);
        }
        
        return maxTime;
    }
};
14.	All paths from source to target
#include <vector>

using namespace std;

class Solution {
public:
    void dfs(vector<vector<int>>& graph, int node, vector<int>& path, vector<vector<int>>& result) {
        path.push_back(node);
        
        if (node == graph.size() - 1) {
            result.push_back(path);
        } else {
            for (int neighbor : graph[node]) {
                dfs(graph, neighbor, path, result);
            }
        }
        
        path.pop_back(); // Backtrack
    }

    vector<vector<int>> allPathsSourceTarget(vector<vector<int>>& graph) {
        vector<vector<int>> result;
        vector<int> path;
        dfs(graph, 0, path, result);
        return result;
    }
};
15.	Redundant Connection
#include <vector>

using namespace std;

class Solution {
public:
    int find(int x, vector<int>& parent) {
        if (parent[x] != x) {
            parent[x] = find(parent[x], parent);
        }
        return parent[x];
    }

    void unionSets(int x, int y, vector<int>& parent, vector<int>& rank) {
        int rootX = find(x, parent);
        int rootY = find(y, parent);
        
        if (rootX != rootY) {
            if (rank[rootX] > rank[rootY]) {
                parent[rootY] = rootX;
            } else if (rank[rootX] < rank[rootY]) {
                parent[rootX] = rootY;
            } else {
                parent[rootY] = rootX;
                rank[rootX]++;
            }
        }
    }

    vector<int> findRedundantConnection(vector<vector<int>>& edges) {
        int n = edges.size();
        vector<int> parent(n + 1), rank(n + 1, 0);

        for (int i = 1; i <= n; ++i) {
            parent[i] = i;
        }

        for (auto& edge : edges) {
            int u = edge[0], v = edge[1];
            if (find(u, parent) == find(v, parent)) {
                return edge; // Redundant edge found
            }
            unionSets(u, v, parent, rank);
        }

        return {};
    }
};
16.	Shortest Path in Binary Matrix
#include <vector>
#include <queue>

using namespace std;

class Solution {
public:
    int shortestPathBinaryMatrix(vector<vector<int>>& grid) {
        int n = grid.size();
        if (grid[0][0] == 1 || grid[n-1][n-1] == 1) return -1;
        
        // Directions for 8 possible movements
        vector<int> directions = {-1, 0, 1, 0, -1, -1, 1, 1, -1, 1, 1, -1};

        queue<pair<int, int>> q;
        q.push({0, 0});
        grid[0][0] = 1; // Mark visited
        
        int steps = 1;
        while (!q.empty()) {
            int size = q.size();
            for (int i = 0; i < size; ++i) {
                auto [x, y] = q.front();
                q.pop();
                if (x == n-1 && y == n-1) return steps;
                for (int d = 0; d < 8; d += 2) {
                    int nx = x + directions[d], ny = y + directions[d+1];
                    if (nx >= 0 && ny >= 0 && nx < n && ny < n && grid[nx][ny] == 0) {
                        grid[nx][ny] = 1;
                        q.push({nx, ny});
                    }
                }
            }
            steps++;
        }
        
        return -1;
    }
};
17.	Remove Max number of Edges
#include <vector>

using namespace std;

class Solution {
public:
    int find(int x, vector<int>& parent) {
        if (parent[x] != x) {
            parent[x] = find(parent[x], parent);
        }
        return parent[x];
    }

    void unionSets(int x, int y, vector<int>& parent) {
        int rootX = find(x, parent);
        int rootY = find(y, parent);
        if (rootX != rootY) {
            parent[rootX] = rootY;
        }
    }

    int maxNumEdgesToRemove(int n, vector<vector<int>>& edges) {
        vector<int> aliceParent(n + 1), bobParent(n + 1), bothParent(n + 1);
        for (int i = 1; i <= n; ++i) {
            aliceParent[i] = bobParent[i] = bothParent[i] = i;
        }
        
        int edgesRemoved = 0;
        int edgeCount = 0;
        
        // Process Type 3 edges first (both Alice and Bob can traverse)
        for (auto& edge : edges) {
            if (edge[0] == 3) {
                if (find(edge[1], bothParent) != find(edge[2], bothParent)) {
                    unionSets(edge[1], edge[2], bothParent);
                } else {
                    edgesRemoved++;
                }
                edgeCount++;
            }
        }

        // Process Type 1 edges (Alice can traverse)
        for (auto& edge : edges) {
            if (edge[0] == 1) {
                if (find(edge[1], aliceParent) != find(edge[2], aliceParent)) {
                    unionSets(edge[1], edge[2], aliceParent);
                } else {
                    edgesRemoved++;
                }
                edgeCount++;
            }
        }

        // Process Type 2 edges (Bob can traverse)
        for (auto& edge : edges) {
            if (edge[0] == 2) {
                if (find(edge[1], bobParent) != find(edge[2], bobParent)) {
                    unionSets(edge[1], edge[2], bobParent);
                } else {
                    edgesRemoved++;
                }
                edgeCount++;
            }
        }

        // Check if both Alice and Bob can traverse the graph
        if (find(1, aliceParent) != find(n, aliceParent) || find(1, bobParent) != find(n, bobParent)) {
            return -1;
        }

        return edges.size() - edgeCount; // Max edges removed
    }
};
