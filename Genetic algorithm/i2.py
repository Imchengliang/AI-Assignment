import time
import operator
import math
import random
import copy
import pandas as pd
import numpy as np

class Solution():
    def __init__(self):
        self.nodes_seq = None
        self.obj = None
        self.vehicles = 0
        self.routes = None

class Node():
    def __init__(self):
        self.id = 0
        self.seq_no = 0
        self.x_coord = 0
        self.y_coord = 0
        self.demand = 0

class Model():
    def __init__(self):
        self.best_sol = None
        self.node_list = []
        self.sol_list = []
        self.parents = []
        self.children = []
        self.node_seq_no_list = []
        self.depot = None
        self.number_of_nodes = 0
        self.vehicle_cap = 0
        self.pc = 0
        self.pm = 0
        self.n_select = 0
        self.pop_size = 0

def read_file(filepath, model):
    node_seq_no = -1
    df = pd.read_excel(filepath)
    for i in range(df.shape[0]):
        node = Node()
        node.seq_no = node_seq_no
        node.id = df['Node'][i]
        node.x_coord = df['x-coordinate'][i]
        node.y_coord = df['y-coordinate'][i]
        node.demand = df['Demand'][i]
        
        if df['Demand'][i] == 0:
            model.depot = node
        else:
            model.node_list.append(node)
            model.node_seq_no_list.append(node_seq_no)
        node_seq_no += 1

    model.number_of_nodes = len(model.node_list)

def gen_initial(model):
    for i in range(model.pop_size):
        nodes_seq = model.node_seq_no_list
        seed = int(random.randint(0, 100))
        random.seed(seed)
        random.shuffle(nodes_seq)
        sol = Solution()
        sol.nodes_seq = nodes_seq
        model.sol_list.append(sol)
        
def split_routes(nodes_seq, model):
    num_vehicle = 0
    route, vehicle_routes = [], []
    remained_cap = model.vehicle_cap
    for node_no in nodes_seq:
        if remained_cap - model.node_list[node_no].demand >= 0:
            route.append(node_no)
            remained_cap -= model.node_list[node_no].demand
        else:
            vehicle_routes.append(route)
            route = [node_no]
            num_vehicle += 1
            remained_cap = model.vehicle_cap - model.node_list[node_no].demand
    vehicle_routes.append(route)
    return num_vehicle, vehicle_routes
    

def cal_distance(route, model):
    distance = 0
    depot = model.depot
    for i in range(len(route)-1):
        from_node = model.node_list[route[i]]
        to_node = model.node_list[route[i+1]]
        distance += math.sqrt((from_node.x_coord-to_node.x_coord)**2+(from_node.y_coord-to_node.y_coord)**2)
    first_node = model.node_list[route[0]]
    last_node = model.node_list[route[-1]]
    distance += math.sqrt((depot.x_coord-first_node.x_coord)**2+(depot.y_coord-first_node.y_coord)**2)
    distance += math.sqrt((depot.x_coord-last_node.x_coord)**2+(depot.y_coord - last_node.y_coord)**2)
    return distance

def cal_fitness(model):
    for sol in model.sol_list:
        nodes_seq = sol.nodes_seq
        num_vehicle, vehicle_routes = split_routes(nodes_seq, model) 
        if num_vehicle > 9:
            model.sol_list.remove(sol)
            break    
        distance = 0
        for route in vehicle_routes:
            distance += cal_distance(route, model)
        sol.obj = distance
        sol.routes = vehicle_routes
        if sol.obj < model.best_sol.obj:
            model.best_sol = sol

# Binary tournament
def nature_selection(model):
    model.parents = []
    num_list = [j for j in range(len(model.sol_list))]
    for i in range(model.n_select):
        if len(num_list) > 2:
            f1_index = random.choice(num_list)
            num_list.remove(f1_index)
            f2_index = random.choice(num_list)
            num_list.remove(f2_index)
        else:
            new_list = [j for j in range(len(model.sol_list))]
            f1_index = random.choice(new_list)
            new_list.remove(f1_index)
            f2_index = random.choice(new_list)
            new_list.remove(f2_index)

        if model.sol_list[f1_index].obj > model.sol_list[f2_index].obj:
            model.parents.append(model.sol_list[f2_index])
        else:
            model.parents.append(model.sol_list[f1_index])
    
# Order Crossover (OX)
def cross_over(model):
    model.children, model.sol_list = [], []
    num_list = [j for j in range(len(model.parents))]
    j = 0

    for male in model.parents:
        num_list.remove(j)
        j += 1
        if len(num_list) < 1:
            m_index = random.choice([j for j in range(len(model.parents))])
        else:
            m_index = random.choice(num_list)

        female = copy.deepcopy(model.parents[m_index])
        if random.uniform(0, 1) < model.pc:
            left_index = int(random.randint(0, model.number_of_nodes-5))
            right_index = int(random.randint(left_index+1, model.number_of_nodes-1))
            male_gen_1 = []
            male_gen = male.nodes_seq[left_index: right_index+1]
            male_gen_2 = []
            female_gen_1 = []
            female_gen = female.nodes_seq[left_index: right_index+1]
            female_gen_2 = []

            for index in range(model.number_of_nodes):
                if len(male_gen_1) < left_index:
                    if female.nodes_seq[index] not in male_gen:
                        male_gen_1.append(female.nodes_seq[index])
                else:
                    if female.nodes_seq[index] not in male_gen:
                        male_gen_2.append(female.nodes_seq[index])
            
            for index in range(model.number_of_nodes):
                if len(female_gen_1) < left_index:
                    if male.nodes_seq[index] not in female_gen:
                        female_gen_1.append(male.nodes_seq[index])
                else:
                    if male.nodes_seq[index] not in female_gen:
                        female_gen_2.append(male.nodes_seq[index])
            
            male_gen_1.extend(male_gen)
            male_gen_1.extend(male_gen_2)
            male.nodes_seq = male_gen_1
            
            female_gen_1.extend(female_gen)
            female_gen_1.extend(female_gen_2)
            female.nodes_seq = female_gen_1
            
            mutation(model, male)
            mutation(model, female)

        else:
            mutation(model, male)
            mutation(model, female) 
        
# Mutation
def mutation(model, child):
    num = random.choice([1, 2, 3])
    if random.uniform(0, 1) < model.pm:
        if num == 1:
            m1_index = random.randint(0, model.number_of_nodes-3)
            m2_index = random.randint(m1_index+2, model.number_of_nodes-1)
            node1 = child.nodes_seq[m1_index]
            child.nodes_seq[m1_index] = child.nodes_seq[m2_index]
            child.nodes_seq[m2_index] = node1
            model.sol_list.append(child)

        elif num == 2:
            m1_index = random.randint(0, model.number_of_nodes-3)
            m2_index = random.randint(m1_index+2, model.number_of_nodes-1)
            new = child.nodes_seq[m1_index]
            del child.nodes_seq[m1_index]
            child.nodes_seq.insert(m2_index, new)
            model.sol_list.append(child)
        
        elif num == 3:
            left_index = int(random.randint(0, model.number_of_nodes-4))
            right_index = int(random.randint(left_index+2, model.number_of_nodes-1))
            old = child.nodes_seq[left_index: right_index+1]
            old1 = child.nodes_seq[0: left_index]
            old2 = child.nodes_seq[right_index+1:]
            new = old[::-1]
            old1.extend(new)
            old1.extend(old2)
            child.nodes_seq = old1
            model.sol_list.append(child)

    else:
        model.sol_list.append(child)

def run(filepath, generations, pc, pm, pop_size, n_select, v_cap):
    model = Model()
    model.vehicle_cap = v_cap
    model.pc = pc
    model.pm = pm
    model.pop_size = pop_size
    model.n_select = n_select

    read_file(filepath, model)
    gen_initial(model)
    model.best_sol = Solution()
    model.best_sol.obj = float('inf')
    start_time = time.time()
    for gen in range(generations):
        cal_fitness(model)
        nature_selection(model)
        cross_over(model)
        print("%s/%s， best obj: %s" % (gen, generations, model.best_sol.obj))
    print('Time:', round(time.time() - start_time, 0))
    #outPut(model)

if __name__=='__main__':
    #file = '2.xlsx'
    run(filepath='2.xlsx', generations=1000, pc=0.6, pm=0.2, pop_size=200, n_select=100, v_cap=100)
    
