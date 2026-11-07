"""
tree_search_solution.py
Cài đặt các thuật toán tìm kiếm trên không gian trạng thái (State Space Search)
cho bài toán Maze với bảng màu và ký hiệu chuẩn xác theo kết quả notebook.
"""

import heapq
import itertools
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import maze_helper as mh

# ==========================================
# Cấu hình thứ tự hướng đi (Directions)
# ==========================================
DIRECTIONS = ['N', 'E', 'S', 'W']
RANDOM_ORDER = False


def set_order(order=None, random_order=False, random=None):
    """
    Thiết lập thứ tự duyệt hướng đi:
    - order: chuỗi hoặc list, ví dụ: "NESW", "SENW", ['N', 'E', 'S', 'W']
    - random / random_order: True nếu muốn thứ tự ngẫu nhiên ở mỗi bước
    """
    global DIRECTIONS, RANDOM_ORDER
    if random is not None:
        random_order = random
    RANDOM_ORDER = random_order

    if order is not None:
        if isinstance(order, str):
            DIRECTIONS = list(order.upper())
        else:
            DIRECTIONS = list(order)

    if RANDOM_ORDER:
        print("Directions are checked at every step in random order.")
    else:
        print(f"Directions are checked in the order {DIRECTIONS}")


def get_current_directions():
    global DIRECTIONS, RANDOM_ORDER
    if RANDOM_ORDER:
        dirs = list(DIRECTIONS)
        random.shuffle(dirs)
        return dirs
    return list(DIRECTIONS)


MOVE_DELTAS = {
    'N': (-1, 0),
    'S': (1, 0),
    'E': (0, 1),
    'W': (0, -1)
}

# ==========================================
# Cấu trúc Node trên Cây Tìm Kiếm
# ==========================================


class Node:
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state = state          # Tọa độ (row, col)
        self.parent = parent
        self.action = action        # Hướng đi: 'N', 'E', 'S', 'W'
        self.path_cost = path_cost  # g(n)

    def __repr__(self):
        return f"Node(state={self.state}, action={self.action}, cost={self.path_cost})"


def get_successors(maze, state, reverse_order=False):
    """Trả về danh sách (action, next_state) hợp lệ từ vị trí hiện tại."""
    dirs = get_current_directions()
    if reverse_order:
        dirs = list(reversed(dirs))

    successors = []
    r, c = state
    rows, cols = maze.shape
    for action in dirs:
        dr, dc = MOVE_DELTAS[action]
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            if maze[nr, nc] != 'X':
                successors.append((action, (nr, nc)))
    return successors


def extract_solution(node):
    """Truy vết nghiệm từ node đích về node gốc."""
    path = []
    actions = []
    curr = node
    while curr is not None:
        path.append(curr.state)
        if curr.action is not None:
            actions.append(curr.action)
        curr = curr.parent
    path.reverse()
    actions.reverse()
    return path, actions

# ==========================================
# Hàm Heuristic
# ==========================================


def manhattan(state, goal):
    """Khoảng cách Manhattan"""
    return abs(state[0] - goal[0]) + abs(state[1] - goal[1])


def euclidean(state, goal):
    """Khoảng cách Euclidean"""
    return math.sqrt((state[0] - goal[0])**2 + (state[1] - goal[1])**2)


heuristic = manhattan


def min_index(values):
    """Trả về chỉ số của phần tử có giá trị nhỏ nhất đầu tiên."""
    return int(np.argmin(values))

# ==========================================
# Helper tạo frame ma trận cho hiển thị & Animation
# ==========================================


def create_maze_frame(base_maze, reached_states, dead_ends, path_states, start, goal):
    """
    Quy ước màu chuẩn xác:
    0: ' ' = Trắng (chưa khám phá)
    1: 'X' = Đen (tường)
    2: 'S' = Xanh dương (Start)
    3: 'G' = Xanh lá (Goal)
    4: '.' = Đỏ (đã duyệt / reached)
    5: 'F' = Xám (ngõ cụt / pruned DFS)
    6: 'P' = Cam (đường đi kết quả / Path)
    """
    frame = np.copy(base_maze)
    # Các ô đã duyệt gán nhãn '.' -> màu đỏ (mã 4)
    for s in reached_states:
        if frame[s] not in ['X', 'S', 'G']:
            frame[s] = '.'
    # Các ô ngõ cụt/backtrack gán nhãn 'F' -> màu xám (mã 5)
    for s in dead_ends:
        if frame[s] not in ['X', 'S', 'G']:
            frame[s] = 'F'
    # Các ô trên đường đi kết quả gán nhãn 'P' -> màu cam (mã 6)
    for s in path_states:
        if frame[s] not in ['S', 'G']:
            frame[s] = 'P'

    frame[start] = 'S'
    frame[goal] = 'G'
    return frame

# ==========================================
# Thuật toán Best-First Search (BFS, GBFS, A*, Weighted A*)
# ==========================================


def best_first_search(maze, strategy="BFS", W=1.0, debug=False, vis=False, animation=False):
    start = mh.find_pos(maze, "S")
    goal = mh.find_pos(maze, "G")

    start_node = Node(state=start, parent=None, action=None, path_cost=0)

    if start == goal:
        path, actions = extract_solution(start_node)
        final_frame = create_maze_frame(
            maze, {start}, set(), path, start, goal)
        return {
            'path': path,
            'actions': actions,
            'reached': {start},
            'dead_ends': set(),
            'maze_anim': [final_frame]
        }

    reached = {start: start_node.path_cost}
    counter = itertools.count()
    frontier = []
    maze_anim = []

    def get_priority(node, count):
        g = node.path_cost
        h = heuristic(node.state, goal)
        if strategy == "BFS":
            return (g, count)
        elif strategy == "GBFS":
            return (h, -count)
        elif strategy == "A*":
            f = g + W * h
            return (f, -count)
        else:
            return (g, count)

    c_init = next(counter)
    prio = get_priority(start_node, c_init)
    heapq.heappush(frontier, (prio, c_init, start_node))

    if animation:
        maze_anim.append(create_maze_frame(
            maze, set(reached.keys()), set(), [], start, goal))

    goal_node = None

    while frontier:
        _, _, current_node = heapq.heappop(frontier)
        current_state = current_node.state

        if current_state == goal:
            goal_node = current_node
            break

        for action, next_state in get_successors(maze, current_state):
            cost = current_node.path_cost + 1
            if next_state not in reached or cost < reached[next_state]:
                reached[next_state] = cost
                child_node = Node(
                    state=next_state, parent=current_node, action=action, path_cost=cost)
                c_idx = next(counter)
                p = get_priority(child_node, c_idx)
                heapq.heappush(frontier, (p, c_idx, child_node))

        if animation:
            maze_anim.append(create_maze_frame(
                maze, set(reached.keys()), set(), [], start, goal))

    if goal_node:
        path, actions = extract_solution(goal_node)
    else:
        path, actions = None, None

    if animation:
        maze_anim.append(create_maze_frame(
            maze, set(reached.keys()), set(), path if path else [], start, goal))

    if not animation:
        if path is not None:
            print(f"Path length: {len(actions)}")
        print(f"Reached squares: {len(reached)}")

    return {
        'path': path,
        'actions': actions,
        'reached': set(reached.keys()),
        'dead_ends': set(),
        'maze_anim': maze_anim
    }

# ==========================================
# Thuật toán Depth-First Search (DFS) & DLS
# ==========================================


def DFS(maze, limit=None, frontier_option=None, max_tries=100000,
        check_cycle=True, vis=False, debug_reached=False, animation=False):
    start = mh.find_pos(maze, "S")
    goal = mh.find_pos(maze, "G")

    start_node = Node(state=start, parent=None, action=None, path_cost=0)

    if start == goal:
        path, actions = extract_solution(start_node)
        final_frame = create_maze_frame(
            maze, {start}, set(), path, start, goal)
        return {
            'path': path,
            'actions': actions,
            'reached': {start},
            'dead_ends': set(),
            'maze_anim': [final_frame]
        }

    stack = [start_node]
    reached = {start}
    explored_all = {start}
    tries = 0
    goal_node = None
    maze_anim = []

    if animation:
        maze_anim.append(create_maze_frame(
            maze, reached, set(), [], start, goal))

    while stack and tries < max_tries:
        tries += 1
        current_node = stack.pop()
        current_state = current_node.state

        if current_state == goal:
            goal_node = current_node
            break

        if limit is not None and current_node.path_cost >= limit:
            continue

        for action, next_state in get_successors(maze, current_state, reverse_order=True):
            if check_cycle:
                p = current_node
                cycle = False
                while p is not None:
                    if p.state == next_state:
                        cycle = True
                        break
                    p = p.parent
                if cycle:
                    continue

            reached.add(next_state)
            explored_all.add(next_state)
            child_node = Node(state=next_state, parent=current_node,
                              action=action, path_cost=current_node.path_cost + 1)
            stack.append(child_node)

        if animation:
            # Các ô trên stack là ô đang xem xét, ô đã từng duyệt nhưng không trên stack hiển thị màu xám
            current_path_nodes = set()
            curr = current_node
            while curr is not None:
                current_path_nodes.add(curr.state)
                curr = curr.parent
            dead_ends = explored_all - current_path_nodes
            maze_anim.append(create_maze_frame(
                maze, current_path_nodes, dead_ends, [], start, goal))

    if goal_node:
        path, actions = extract_solution(goal_node)
    else:
        path, actions = None, None

    # Xác định các ô ngõ cụt (dead-ends) sau khi duyệt xong
    path_set = set(path) if path else set()
    dead_ends = explored_all - path_set

    if animation:
        maze_anim.append(create_maze_frame(
            maze, path_set, dead_ends, path if path else [], start, goal))

    if not animation:
        if path is not None:
            print(f"Path length: {len(actions)}")
        if debug_reached:
            print(f"Reached squares: {len(explored_all)}")

    return {
        'path': path,
        'actions': actions,
        'reached': explored_all,
        'dead_ends': dead_ends,
        'maze_anim': maze_anim
    }

# ==========================================
# Thuật toán Iterative Deepening Search (IDS)
# ==========================================


def IDS(maze, frontier_option=None, max_tries=100000, animation=False):
    depth = 0
    while depth < max_tries:
        res = DFS(maze, limit=depth, check_cycle=True, max_tries=max_tries,
                  debug_reached=False, animation=animation)
        if res['path'] is not None:
            res['reached'] = set()
            return res
        depth += 1
    return {
        'path': None,
        'actions': None,
        'reached': set(),
        'dead_ends': set(),
        'maze_anim': []
    }

# ==========================================
# Hàm hiển thị kết quả tìm kiếm (show_path)
# ==========================================


def show_path(maze, result, fontsize=10):
    """
    Hiển thị đúng màu sắc và nhãn theo quy ước của notebook:
    - Trắng: Ô trống chưa duyệt
    - Đen: Tường ('X')
    - Xanh dương: Bắt đầu ('S')
    - Xanh lá: Đích ('G')
    - Đỏ: Các ô đã duyệt ('reached')
    - Xám: Các ngõ cụt đã quay lui / bỏ qua (dead-ends)
    - Cam: Đường đi nghiệm ('P' / path)
    """
    # Thứ tự màu: [0: white, 1: black, 2: blue, 3: green, 4: red, 5: gray, 6: orange]
    cmap = colors.ListedColormap(
        ['white', 'black', 'blue', 'green', 'red', 'gray', 'orange'])

    vis_maze = np.copy(maze)
    start = mh.find_pos(maze, 'S')
    goal = mh.find_pos(maze, 'G')

    reached = result.get('reached', set())
    dead_ends = result.get('dead_ends', set())
    path = result.get('path', None)

    # 1. Các ô đã duyệt: Đỏ (mã 4)
    for r, c in reached:
        if vis_maze[r, c] not in ['X', 'S', 'G']:
            vis_maze[r, c] = '.'

    # 2. Các ô ngõ cụt: Xám (mã 5)
    for r, c in dead_ends:
        if vis_maze[r, c] not in ['X', 'S', 'G']:
            vis_maze[r, c] = 'F'

    # 3. Đường đi nghiệm: Cam (mã 6)
    if path:
        for r, c in path:
            if vis_maze[r, c] not in ['S', 'G']:
                vis_maze[r, c] = 'P'

    # Giữ nguyên Start và Goal
    vis_maze[start] = 'S'
    vis_maze[goal] = 'G'

    # Ánh xạ thành mảng số nguyên tương ứng Colormap
    maze_numeric = np.zeros(vis_maze.shape, dtype=int)
    maze_numeric[vis_maze == ' '] = 0  # Trắng
    maze_numeric[vis_maze == 'X'] = 1  # Đen (Tường)
    maze_numeric[vis_maze == 'S'] = 2  # Xanh dương (Start)
    maze_numeric[vis_maze == 'G'] = 3  # Xanh lá (Goal)
    maze_numeric[vis_maze == '.'] = 4  # Đỏ (Đã duyệt)
    maze_numeric[vis_maze == 'F'] = 5  # Xám (Ngõ cụt)
    maze_numeric[vis_maze == 'P'] = 6  # Cam (Đường đi nghiệm)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.imshow(maze_numeric, cmap=cmap, norm=colors.BoundaryNorm(
        list(range(cmap.N + 1)), cmap.N))

    # Chữ 'S' và 'G' màu trắng ở giữa ô
    ax.text(start[1], start[0], "S", fontsize=fontsize, color="white",
            horizontalalignment='center', verticalalignment='center', fontweight='bold')
    ax.text(goal[1], goal[0], "G", fontsize=fontsize, color="white",
            horizontalalignment='center', verticalalignment='center', fontweight='bold')

    plt.show()


# Kế thừa hàm animate_maze từ maze_helper
animate_maze = mh.animate_maze
