import sys
import ast
import copy
import heapq
import math
import numpy as np


def inputData():
    inData = sys.argv

    # read config file
    data = open(inData[1], 'r')
    lines = data.readlines()
    for i in range(len(lines)):
        lines[i] = lines[i][:-1]

    # read keyword of the question type
    question = lines[0]

    heuristic = 'euc'
    if (len(inData)) > 3:
        heuristic = inData[3]

    if question == 'jugs':
        graph, startPoint, endPoint, restrain = read_jugs(lines)
        if inData[2] == 'dfs':
            nodesNo, listLen, changedStruct, parent = dfs(question, graph, startPoint, endPoint, restrain)
            way_to_goal = outputAnsj(parent, endPoint)
            outputAnalyse(nodesNo, listLen, way_to_goal)
        if inData[2] == 'bfs':
            nodesNo, listLen, changedStruct, parent = bfs(question, graph, startPoint, endPoint, restrain)
            way_to_goal = outputAnsj(parent, endPoint)
            outputAnalyse(nodesNo, listLen, way_to_goal)
        if inData[2] == 'unicost':
            nodesNo, listLen, changedStruct, parent, distance = unicost(question, graph, startPoint, endPoint, restrain)
            way_to_goal = outputAnsj(parent, endPoint)
            outputAnalyse(nodesNo, listLen, way_to_goal)
        if inData[2] == 'deep':
            judge, nodesNo, listLen, changedStruct, parent = deep_limited_dfs(question, graph, startPoint, endPoint,
                                                                              restrain, 100)
            way_to_goal = outputAnsj(parent, endPoint)
            outputAnalyse(nodesNo, listLen, way_to_goal)
        if inData[2] == 'iddfs':
            nodesNo, listLen, changedStruct, parent = iddfs(question, graph, startPoint, endPoint, restrain)
            way_to_goal = outputAnsj(parent, endPoint)
            outputAnalyse(nodesNo, listLen, way_to_goal)
        if inData[2] == 'greedy':
            nodesNo, listLen, changedStruct, parent = greedy(question, graph, startPoint, endPoint, restrain, 'heu')
            way_to_goal = outputAnsj(parent, endPoint)
            outputAnalyse(nodesNo, listLen, way_to_goal)
        if inData[2] == 'astar':
            nodesNo, listLen, changedStruct, parent = astar(question, graph, startPoint, endPoint, restrain, heuristic)
            way_to_goal = outputAnsj(parent, endPoint)
            outputAnalyse(nodesNo, listLen, way_to_goal)
        if inData[2] == 'idastar':
            nodesNo, listLen, changedStruct, parent = idastar(question, graph, startPoint, endPoint, restrain,
                                                              heuristic)
            way_to_goal = outputAnsj(parent, endPoint)
            outputAnalyse(nodesNo, listLen, way_to_goal)

    if question == 'cities':
        graph, startPoint, endPoint = read_cities(lines)
        if inData[2] == 'dfs':
            nodesNo, listLen, changedStruct, parent = dfs(question, graph, startPoint, endPoint, 0)
            way_to_goal = outputAnsc(changedStruct, endPoint)
            outputAnalyse(nodesNo, listLen, way_to_goal)
        if inData[2] == 'bfs':
            nodesNo, listLen, changedStruct, parent = bfs(question, graph, startPoint, endPoint, 0)
            way_to_goal = outputAnsc(changedStruct, endPoint)
            outputAnalyse(nodesNo, listLen, way_to_goal)
        if inData[2] == 'unicost':
            nodesNo, listLen, changedStruct, parent, distance = unicost(question, graph, startPoint, endPoint, 0)
            way_to_goal = outputAnsj(parent, endPoint)
            outputAnalyse(nodesNo, listLen, way_to_goal)
            print("least cost to each node :", distance)
        if inData[2] == 'deep':
            judge, nodesNo, listLen, changedStruct, parent = deep_limited_dfs(question, graph, startPoint, endPoint, 0,
                                                                              100)
            way_to_goal = outputAnsc(changedStruct, endPoint)
            outputAnalyse(nodesNo, listLen, way_to_goal)
        if inData[2] == 'iddfs':
            nodesNo, listLen, changedStruct, parent = iddfs(question, graph, startPoint, endPoint, 0)
            way_to_goal = outputAnsc(changedStruct, endPoint)
            outputAnalyse(nodesNo, listLen, way_to_goal)
        if inData[2] == 'greedy':
            nodesNo, listLen, changedStruct, parent = greedy(question, graph, startPoint, endPoint, 0, 'euc')
            way_to_goal = outputAnsc(changedStruct, endPoint)
            outputAnalyse(nodesNo, listLen, way_to_goal)
        if inData[2] == 'astar':
            nodesNo, listLen, changedStruct, parent = astar(question, graph, startPoint, endPoint, 0, heuristic)
            way_to_goal = outputAnsc(changedStruct, endPoint)
            outputAnalyse(nodesNo, listLen, way_to_goal)
        if inData[2] == 'idastar':
            nodesNo, listLen, changedStruct, parent = idastar(question, graph, startPoint, endPoint, 0, heuristic)
            way_to_goal = outputAnsc(changedStruct, endPoint)
            outputAnalyse(nodesNo, listLen, way_to_goal)

    if question == 'pancakes':
        graph, startPoint, endPoint, restrain = read_pancakes(lines)
        if inData[2] == 'dfs':
            nodesNo, listLen, changedStruct, parent = dfs(question, graph, startPoint, endPoint, restrain)
            way_to_goal = outputAnsj(parent, endPoint)
            outputAnalyse(nodesNo, listLen, way_to_goal)
        if inData[2] == 'bfs':
            nodesNo, listLen, changedStruct, parent = bfs(question, graph, startPoint, endPoint, restrain)
            way_to_goal = outputAnsj(parent, endPoint)
            outputAnalyse(nodesNo, listLen, way_to_goal)
        if inData[2] == 'unicost':
            nodesNo, listLen, changedStruct, parent, distance = unicost(question, graph, startPoint, endPoint, restrain)
            way_to_goal = outputAnsj(parent, endPoint)
            outputAnalyse(nodesNo, listLen, way_to_goal)
        if inData[2] == 'deep':
            judge, nodesNo, listLen, changedStruct, parent = deep_limited_dfs(question, graph, startPoint, endPoint,
                                                                              restrain, 100)
            way_to_goal = outputAnsj(parent, endPoint)
            outputAnalyse(nodesNo, listLen, way_to_goal)
        if inData[2] == 'iddfs':
            nodesNo, listLen, changedStruct, parent = iddfs(question, graph, startPoint, endPoint, restrain)
            way_to_goal = outputAnsj(parent, endPoint)
            outputAnalyse(nodesNo, listLen, way_to_goal)
        if inData[2] == 'greedy':
            nodesNo, listLen, changedStruct, parent = greedy(question, graph, startPoint, endPoint, restrain, 'heu')
            way_to_goal = outputAnsj(parent, endPoint)
            outputAnalyse(nodesNo, listLen, way_to_goal)
        if inData[2] == 'astar':
            nodesNo, listLen, changedStruct, parent = astar(question, graph, startPoint, endPoint, restrain, heuristic)
            way_to_goal = outputAnsj(parent, endPoint)
            outputAnalyse(nodesNo, listLen, way_to_goal)
        if inData[2] == 'idastar':
            nodesNo, listLen, changedStruct, parent = idastar(question, graph, startPoint, endPoint, restrain,
                                                              heuristic)
            way_to_goal = outputAnsj(parent, endPoint)
            outputAnalyse(nodesNo, listLen, way_to_goal)


def read_cities(lines):
    cities = ast.literal_eval(lines[1])
    startPoint = lines[2][1:-1]  # delete '"'
    endPoint = lines[3][1:-1]
    roads = lines[4:]
    road_on_map = []
    for i in range(len(roads)):
        road_on_map.append(ast.literal_eval(roads[i]))
    # store cities' name, location, parent, nearby cities, cost to nearby , action
    graph = {}
    for city in cities:
        graph[city[0]] = []
        graph[city[0]].append((city[1], city[2]))
        graph[city[0]].append(None)
    for path in road_on_map:
        graph[path[0]].append((path[1], path[2], (path[0], path[1])))
        graph[path[1]].append((path[0], path[2], (path[1], path[0])))
    return graph, startPoint, endPoint


def read_jugs(lines):
    jugsLimit = ast.literal_eval(lines[1])
    jugsWater = ast.literal_eval(lines[2])
    jugsGoal = ast.literal_eval(lines[3])
    graph = {}
    info = [jugsLimit, None]
    info += generate_child_to_graph('jugs', jugsWater, graph, jugsLimit, 1)
    graph[jugsWater] = info
    return graph, jugsWater, jugsGoal, jugsLimit


def read_pancakes(lines):
    initPancakes = tuple(ast.literal_eval(lines[1]))
    goalPancakes = tuple(ast.literal_eval(lines[2]))
    graph = {}
    info = [goalPancakes, None]
    number = len(initPancakes)
    print(number, "number")
    info += generate_child_to_graph('pancakes', initPancakes, graph, len(goalPancakes), 1)
    graph[tuple(initPancakes)] = info
    return graph, initPancakes, goalPancakes, len(goalPancakes)


def generate_child_to_graph(question, current, graph, restrains, cost):
    if question == 'jugs':
        restrain = list(restrains)
        child = []
        for i in range(len(restrain)):
            # empty i
            if current[i] != 0:
                temp = list(current)
                temp[i] = 0
                child.append([tuple(temp), cost, (current, tuple(temp))])
            # fill i
            if current[i] == 0:
                temp = list(current)
                temp[i] = restrain[i]
                child.append([tuple(temp), cost, (current, tuple(temp))])
            for j in range(len(current)):
                temp = list(current)
                if j != i:
                    # pour j to i
                    if current[j] != 0:
                        if current[j] > (restrain[i] - current[i]):
                            temp[i] = restrain[i]
                            temp[j] = current[j] - (restrain[i] - current[i])
                            child.append([tuple(temp), cost, (current, tuple(temp))])
                        else:
                            temp[i] = current[i] + current[j]
                            temp[j] = 0
                            child.append([tuple(temp), cost, (current, tuple(temp))])
                    # pour i to j
                    if current[i] != 0:
                        if current[i] > (restrain[j] - current[j]):
                            temp[j] = restrain[j]
                            temp[i] = current[i] - (restrain[j] - current[j])
                            child.append([tuple(temp), cost, (current, tuple(temp))])
                        else:
                            temp[j] = current[j] + current[i]
                            temp[i] = 0
                            child.append([tuple(temp), cost, (current, tuple(temp))])
        childs = [restrain, None]
        childs += child
        graph.update({current: childs})
        return child

    if question == 'pancakes':
        child = []
        for i in range(1, restrains + 1):
            temp = current
            next = flip(temp, i)
            child.append([tuple(next), cost, (tuple(current), tuple(next))])
        childs = [restrains, None]
        childs += child
        graph.update({tuple(current): childs})
        return child


def flip(s, flipPoint):
    stack = []
    flipped = []
    temp = list(copy.deepcopy(s))
    for i in range(flipPoint):
        last = temp.pop(0)
        stack.append(-last)
    for i in range(flipPoint):
        flipped.append(stack.pop(-1))
    return flipped + temp


# print the Time, Space , how to get the goal state
def outputAnalyse(time, space, path):
    print("path to the goal:", path)
    print("Time (how much nodes we have created): ", time)
    print("Space (how big the frontier list grew to): ", space)


# find the way to the goal state by analyse record
def outputAnsc(changedStruct, g):
    path_from_end_to_start = []
    path_from_start_to_end = []
    path_from_end_to_start.append(g)
    v = changedStruct[g][1]
    while v != None:
        path_from_end_to_start.append(v)
        v = changedStruct[v][1]
    for i in range(len(path_from_end_to_start)):
        path_from_start_to_end.append(path_from_end_to_start.pop())
    return path_from_start_to_end


def outputAnsj(parent, g):
    path_from_end_to_start = []
    path_from_start_to_end = []
    path_from_end_to_start.append(g)
    v = parent[g]
    while v != None:
        path_from_end_to_start.append(v)
        v = parent[v]
    for i in range(len(path_from_end_to_start)):
        path_from_start_to_end.append(path_from_end_to_start.pop())
    return path_from_start_to_end


def testGoal(start, goal):
    if start == goal:
        print("no nead to search! We already know the anwser.")
    return


def dfs(question, graph, s, g, restrain):
    nodeNo = 1
    parent = {s: None}
    stack = list()
    stack.append(s)
    exploded = set()
    exploded.add(s)
    space = len(exploded)
    testGoal(s, g)
    while (len(stack)) > 0:
        space = len(exploded)
        vertex = stack.pop()
        if vertex == g:
            return nodeNo, space, graph, parent
        if (question == 'jugs') or (question == 'pancakes'):
            generate_child_to_graph(question, vertex, graph, restrain, 1)
        nodes = []
        for i in range(2, len(graph[vertex])):
            nodes.append(list(graph[vertex][i]))
            nodeNo += 1
        dealedNodes = []
        for undealedNode in nodes:
            dealedNodes.append(undealedNode[0])
        for node in dealedNodes:
            if node not in exploded:
                if (question == 'jugs') or (question == 'pancakes'):
                    generate_child_to_graph(question, node, graph, restrain, 1)
                stack.append(node)
                graph[node][1] = vertex
                parent[node] = vertex
                exploded.add(node)


def bfs(question, graph, s, g, restrain):
    nodeNo = 1
    parent = {s: None}
    queue = list()
    queue.append(s)
    exploded = set()
    exploded.add(s)
    space = len(exploded)
    testGoal(s, g)
    while (len(queue)) > 0:
        space = len(exploded)
        vertex = queue.pop(0)
        if vertex == g:
            return nodeNo, space, graph, parent
        if (question == 'jugs') or (question == 'pancakes'):
            generate_child_to_graph(question, vertex, graph, restrain, 1)
        nodes = []
        for i in range(2, len(graph[vertex])):
            nodes.append(list(graph[vertex][i]))
            nodeNo += 1
        dealedNodes = []
        for undealedNode in nodes:
            dealedNodes.append(undealedNode[0])
        for node in dealedNodes:
            if node not in exploded:
                if (question == 'jugs') or (question == 'pancakes'):
                    generate_child_to_graph(question, node, graph, restrain, 1)
                queue.append(node)
                graph[node][1] = vertex
                parent[node] = vertex
                exploded.add(node)


def init_distance(graph, s):
    distance = {s: 0}
    for vertex in graph:
        if vertex != s:
            distance[vertex] = math.inf
    return distance


def unicost(question, graph, s, g, restrain):
    nodeNo = 1
    parent = {s: None}
    pqueue = []
    heapq.heappush(pqueue, (0, s))
    exploded = set()
    exploded.add(s)
    space = len(pqueue)
    distance = init_distance(graph, s)
    testGoal(s, g)
    while (len(pqueue) > 0):
        space = max(len(pqueue), space)
        pair = heapq.heappop(pqueue)
        dist = pair[0]
        vertex = pair[1]
        if vertex == g:
            return nodeNo, space, graph, parent, distance
        if (question == 'jugs') or (question == 'pancakes'):
            generate_child_to_graph(question, vertex, graph, restrain, 1)
        nodes = []
        for i in range(2, len(graph[vertex])):
            nodes.append(list(graph[vertex][i]))
            nodeNo += 1
        dealedNodes = []
        for undealedNode in nodes:
            dealedNodes.append(undealedNode[0])
        for node in dealedNodes:
            if node not in exploded:
                if (question == 'jugs') or (question == 'pancakes') or (node not in graph.keys()):
                    generate_child_to_graph(question, vertex, graph, restrain, 1)
                for w in nodes:
                    if w[0] == node:
                        if w[0] not in distance.keys():
                            distance[w[0]] = math.inf
                        if dist + w[1] < distance[w[0]]:
                            heapq.heappush(pqueue, (dist + w[1], w[0]))
                            nodeNo += 1
                            graph[w[0][1]] = vertex
                            parent[node] = vertex
                            distance[w[0]] = dist + w[1]


def heuristic_euclidean_distance(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def heuristic_manhattan_distance(a, b):
    return sum(map(lambda i, j: abs(i - j), a, b))


def heuristic_chebyshev_distance(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return np.max(np.abs(x1 - y1), np.abs(x2 - y2))


def heuristic_minkowski_distance(a, b):
    return np.sqrt(np.sum(np.square(a - b)))


def myheuj(next_node, g):
    a = list(next_node)
    b = list(g)
    x1, y1 = a
    x2, y2 = b
    h = abs(x1 - x2) + abs(y1 - y2)
    return h


def greedy(question, graph, s, g, restrain, heu):
    nodeNo = 1
    pqueue = []
    heapq.heappush(pqueue, (0, s))
    exploded = {}
    exploded[s] = 0
    space = len(exploded)
    parent = {s: None}
    while len(pqueue) > 0:
        space = max(len(exploded), space)
        vertex = heapq.heappop(pqueue)[1]
        if vertex == g:
            return nodeNo, space, graph, parent
        if (question == 'jugs') or (question == 'pancakes'):
            generate_child_to_graph(question, vertex, graph, restrain, 1)
        nodes = graph[vertex][2:]
        nodeNo += len(nodes)
        for nextNode in nodes:
            new_cost = exploded[vertex] + nextNode[1]
            if (question == 'jugs') or (question == 'pancakes'):
                if nextNode[0] not in graph.keys():
                    generate_child_to_graph(question, nextNode[0], graph, restrain, 1)
            next_node = graph[nextNode[0]]
            if (nextNode[0] not in exploded) or (new_cost < exploded[nextNode[0]]):
                if question == 'cities':
                    exploded[nextNode[0]] = heuristic_manhattan_distance(graph[g][0], next_node[0])
                if question == 'pancakes':
                    exploded[nextNode[0]] = new_cost
                if question == 'jugs':
                    exploded[nextNode[0]] = myheuj(vertex, g)
                priority = new_cost
                heapq.heappush(pqueue, (priority, nextNode[0]))
                graph[nextNode[0]][1] = vertex
                parent[nextNode[0]] = vertex


def deep_limited_dfs(question, graph, s, g, restrain, depth):
    path = set()
    space = len(path)
    stack = list()
    parent = {s: None}
    stack.append((s, 0))
    stackcheck =[]
    stackcheck.append(s)
    nodeNo = 0
    in_depth=depth
    while len(stack) > 0:
        space = len(path)
        vertex = stack.pop()
        stackcheck.pop()
        if vertex[1] <= in_depth:
            if vertex[0] == g:
                return True, nodeNo, space, graph, parent
            if vertex[0] not in graph.keys() and vertex[1] <= in_depth:
                if (question == 'jugs') or (question == 'pancakes'):
                    generate_child_to_graph(question, vertex[0], graph, restrain, 1)
            nodes = []
            for i in range(2, len(graph[vertex[0]])):
                nodes.append(list(graph[vertex[0]][i]))
                nodeNo += 1
            dealedNodes = []
            for undealedNode in nodes:
                dealedNodes.append(undealedNode[0])
            for node in dealedNodes:
                if node in path:
                    continue
                if node not in stackcheck:
                    if node != vertex[0]:
                        parent[node] = vertex[0]
                    if vertex[1] + 1 <= in_depth:
                        stack.append((node, vertex[1] + 1))
                        stackcheck.append(node)
                        if node in graph.keys():
                            graph[node][1] = vertex[0]
            path.add(vertex[0])
    return False, nodeNo, space, graph, parent


def iddfs(question, graph, s, g, restrain):
    nodeNo = 1
    for i in range(1000):
        judge, nodesNo, listLen, changedStruct, parent = deep_limited_dfs(question, graph, s, g, restrain, i)
        nodeNo += nodesNo
        if judge:
            return nodesNo, listLen, changedStruct, parent

def myheuforp(vertex,g):
    a = sum(abs(list(vertex)))
    b= sum(list(g))
    c = a-b
    return c


def astar(question, graph, s, g, restrain, heu):
    nodeNo = 1
    pqueue = []
    heapq.heappush(pqueue, (0, s))
    exploded = {}
    exploded[s] = 0
    space = len(exploded)
    parent = {s: None}
    while len(pqueue) > 0:
        space = max(len(exploded), space)
        vertex = heapq.heappop(pqueue)[1]
        if vertex == g:
            return nodeNo, space, graph, parent
        if (question == 'jugs') or (question == 'pancakes'):
            generate_child_to_graph(question, vertex, graph, restrain, 1)
        nodes = graph[vertex][2:]
        nodeNo += len(nodes)
        for nextNode in nodes:
            new_cost = exploded[vertex] + nextNode[1]
            if (question == 'jugs') or (question == 'pancakes'):
                if nextNode[0] not in graph.keys():
                    generate_child_to_graph(question, nextNode[0], graph, restrain, 1)
            next_node = graph[nextNode[0]]
            if (nextNode[0] not in exploded) or (new_cost < exploded[nextNode[0]]):
                exploded[nextNode[0]] = new_cost
                if g not in graph.keys():
                    generate_child_to_graph(question, g, graph, restrain, 1)
                if (question == 'jugs') or (question == 'cities'):
                    if heu == 'man':
                        priority = new_cost + heuristic_manhattan_distance(graph[g][0], next_node[0])
                    else:
                        priority = new_cost + heuristic_euclidean_distance(graph[g][0], next_node[0])
                if (question == 'pancakes'):
                    #priority = new_cost+myheuforp(vertex,g)
                    priority = new_cost
                heapq.heappush(pqueue, (priority, nextNode[0]))
                graph[nextNode[0]][1] = vertex
                parent[nextNode[0]] = vertex


def search(question, graph, s, g, restrain, heu, cost_so_far, bound, parent, nodeNo, space, visited, exploded):
    min_node = 99999999999
    vertex = visited[-1]
    f = cost_so_far + heuristic_euclidean_distance(graph[g][0], graph[s][0])
    if vertex not in graph.keys():
        generate_child_to_graph(question, s, graph, restrain, 1)
    if f > bound:
        return f, nodeNo, space, graph, parent
    if vertex == g:
        return True, nodeNo, space, graph, parent
    nodes = graph[vertex][2:]
    nodeNo += len(nodes)
    for nextNode in nodes:
        new_cost = exploded[vertex] + nextNode[1]
        if (question == 'jugs') or (question == 'pancakes'):
            if nextNode[0] not in graph.keys():
                generate_child_to_graph(question, nextNode[0], graph, restrain, 1)
        next_node = graph[nextNode[0]]
        if (nextNode[0] not in exploded) or (new_cost < exploded[nextNode[0]]):
            exploded[nextNode[0]] = new_cost
            if g not in graph.keys():
                generate_child_to_graph(question, g, graph, restrain, 1)
            graph[nextNode[0]][1] = vertex
            if nextNode[0] != vertex:
                parent[nextNode[0]] = vertex
        if nextNode[0] not in visited:
            visited.append(nextNode[0])
            space = max(len(visited), space)
            temp, nodeNo, listLen, changedStruct, parent = search(question, graph, vertex, g, restrain, heu, new_cost,
                                                                  bound, parent, nodeNo, space, visited, exploded)
            if temp == True and type(temp)==type(True):
                return True, nodeNo, space, graph, parent
            if temp < min_node:
                min_node = temp
    visited.pop()
    return min_node, nodeNo, space, graph, parent


def idastar(question, graph, s, g, restrain, heu):
    nodeNo = 1
    if question == 'jugs':
        generate_child_to_graph(question, g, graph, restrain, 1)
        if heu == 'euc':
            bound = heuristic_euclidean_distance(graph[g][0], graph[s][0])
    if question == 'cities':
        if heu == 'euc':
            bound = heuristic_euclidean_distance(graph[g][0], graph[s][0])
    cost_so_far = 0
    exploded = {s: 0}
    visited = list()
    visited.append(s)
    parent = {s: None}
    space = len(exploded)
    testGoal(s, g)
    while (1):
        judge, nodeNo, listLen, changedStruct, parent = search(question, graph, s, g, restrain, heu, cost_so_far, bound,
                                                               parent, nodeNo, space, visited, exploded)
        if (judge == True and type(judge)==type(True)):
            return nodeNo, listLen, changedStruct, parent
        if (judge == 99999999999):
            print("no solution")
            return nodeNo, listLen, changedStruct, parent
        bound = judge


inputData()
