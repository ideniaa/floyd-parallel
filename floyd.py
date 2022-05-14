import sys
import numpy as np
import networkx as nx
from mpi4py import MPI

# Parallel Floyd-Warshall algorithm 2D process
def parallel_fw(adj_mat, nodes):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    row_k = []
    for k in range(nodes):
        root = int(k / (nodes / size))  # rank process that owns row k
        if rank == root:
            for r in range(nodes):
                row_k[r] = adj_mat[(k % (nodes / size) * nodes + r)]
            comm.bcast(row_k, nodes, MPI.INT, root)

            for i in range(nodes / size):
                for j in range(nodes):
                    distance = adj_mat[i * nodes + k] + row_k[j]
                    if distance < adj_mat[i * nodes + j]:
                        adj_mat[i * nodes + j] = distance


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nodes = None

    # read input file
    try:
        file = open("fb.txt", "r")
    except FileNotFoundError:
        print("No file found.")
        sys.exit(1)

    digraph = nx.read_edgelist(file, create_using=nx.DiGraph(), nodetype=int, edgetype=int)
    graph = digraph.to_undirected()
    adj_mat = nx.to_numpy_matrix(digraph)

    size = comm.Get_size()
    nodes = len(adj_mat)

    # start timer
    if rank == 0:
        start = MPI.Wtime()

    parallel_fw(adj_mat, nodes)

    # end timer
    if rank == 0:
        stop = MPI.Wtime()

    print("time: " + str(stop - start))

    MPI.Finalize()
