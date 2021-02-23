# Delay Tolerant Networks
## Group 10

In this project, we imported the algorithms for finding the shortest path between the two furthest nodes in a network.
We have four algorithms: Bellman_Ford, Dijkstra, Floyd_Warshal, Johnson.

We did an optimization based on the properties of the algorithms.

You can find all the codes in the file High_Level_Programming_Group10.ipynb

### The libraries which we used for this project are:
NumPy
networkx
matplotlib
time
math
queue
scipy

NetworkX is a Python library for studying graphs and networks.


Dijkstra algorithm only can have the results on the networks with positive delays, so we used it just for the networks with positive delays. This algorithm is the fastest algorithm among the algorithms for the networks with positive delays.
https://upload.wikimedia.org/wikipedia/commons/thumb/5/57/Dijkstra_Animation.gif/220px-Dijkstra_Animation.gif

Bellman_Ford algorithm only can have the results on the networks with both positive and negative delays, but not on networks with negative cycles. After the Dijkstra algorithm, Bellman_Ford is the fastest algorithm on the networks without negative cycles.

Johnson's algorithm works for all networks with every delay value, but it is slower than Dijkstra and Bellman_ford for positive and negative delays networks.

Floyd_Warshal is the slowest algorithm among the algorithms. Also, it is like Bellman_Ford, so it does not work with the networks with negative cycles.

