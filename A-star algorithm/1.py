import math
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from queue import PriorityQueue
from collections import deque

class Graph:
    def __init__(self, file1):
        # save the length of edges
        graph_edges=[]                
        for i in range(len(file1)):
            graph_edges.append((file1.iloc[i]['from'], file1.iloc[i]['to'], file1.iloc[i]['distance']))   
        
        # use set to record the connected nodes
        self.nodes =set()
        for edge in graph_edges:
            self.nodes.update([edge[0],edge[1]])
        # set up the adjacency list
        self.adjacency_list = {node: set() for node in self.nodes}
        for edge in graph_edges:
            self.adjacency_list[edge[0]].add((edge[1],edge[2]))

    def shortest_path(self, start_node, end_node, file2, w):
        # create a priority queue (the priority is higher when the value is smaller, FIFO)
        frontier = PriorityQueue()
        frontier.put([0, start_node])
        # use dictionary to record the node generation (current node: previous node)
        came_from = dict()
        # use dictionary to save the cost from starting node to current node
        current_distance = dict()
        came_from[start_node] = None
        current_distance[start_node] = 0
        # use dictionary to store the coordinate of each node
        node_xy = dict()
        for i in range(len(file2)):
            node_xy[file2.iloc[i]['node']] = (file2.iloc[i]['x'], file2.iloc[i]['y'])
        
        node_generated = 0
        while not frontier.empty():
            # priority queue uses the smallest value as current node
            current = frontier.get()[1]
            if current == end_node:
                break
            # node generation process
            for next, distance in self.adjacency_list[current]:
                # calculate g(n) = current cost(starting node to previous node) + distance(previous node to current node)
                new_distance = current_distance[current] + distance
                if next not in current_distance or new_distance < current_distance[next]:
                    node_generated += 1
                    current_distance[next] = new_distance
                    # f(n)=g(n)*w+h(n)*(1-w)
                    priority = new_distance*w + Graph.heuristic(self, end_node, next, node_xy)*(1-w)
                    # put the next node into frontier
                    frontier.put([priority, next])
                    # record the generated node
                    came_from[next] = current

        path = deque()
        current_node = end_node
        y_n = 1
        # check if the path is found
        if end_node not in list(came_from.keys()):
            y_n, current_distance[end_node] = 0, 0
            
        else:
            # terminate when the path (except the starting node) is recorded
            while came_from[current_node] is not None:
                path.appendleft(current_node)
                current_node = came_from[current_node]
            
            path.appendleft(start_node)

        return path, current_distance[end_node], node_generated, y_n

    # calculate rectilinear distance
    def heuristic(self, end_node, next, node_xy):
        a = node_xy[end_node][0] - node_xy[next][0]
        b = node_xy[end_node][1] - node_xy[next][1]
        return abs(a)+abs(b)

def A_star(file1, file2, start, end, path, w):
    graph = Graph(file1)
    returned_path, returned_distance, node_generated, y_n = graph.shortest_path(start, end, file2, w)
    if y_n == 1:
        print('from/to: {0} -> {1}'.format(start, end))
        print('node generated: {0}'.format(node_generated))
        print('sequence of the path: {0}'.format(returned_path))
        print('length of path: {0}'.format(returned_distance))
    else:
        print('node generated: {0}'.format(node_generated))
        print('There is no path')

def read_data():
    df = pd.read_csv('100 nodes.csv', header=0, index_col=0)
    df1 = df.drop(df[['x', 'y']], axis=1)
    # write the coordinate data into a dataframe
    x = df['x'].tolist()
    y = df['y'].tolist()
    node = [i for i in range(100)]
    df2 = pd.DataFrame({'node':node, 'x':x, 'y':y})
    
    # write the distance between nodes into a dataframe
    _from, _to, _distance = [], [], []
    for indexes in df1.index: 
        for i in range(len(df1.loc[indexes].values)): 
            if(df1.loc[indexes].values[i] > 0): 
                _from.append(indexes)
                _to.append(i)
                _distance.append(df1.loc[indexes].values[i])
    df3 = pd.DataFrame({'from':_from, 'to':_to, 'distance':_distance})
    
    return df3, df2

def draw_graph(df3):
    G = nx.from_pandas_edgelist(df3, 'from', 'to', create_using=nx.DiGraph())
    nx.draw(G, with_labels=True, node_size=300, alpha=0.7, arrows=True)
    plt.show()

if __name__ == "__main__":
    f1, f2 = read_data()
    #draw_graph(file1)
    for i in range(9):
        weight = float(input("weight: "))
        s_node = int(input("starting node: "))
        e_node = int(input("ending node: "))
        if s_node in [i for i in range(100)] and e_node in [i for i in range(100)]:
            A_star(file1=f1, file2=f2, start=s_node, end=e_node, path=[], w=weight)
        else:
            print('Please input the correct node')