#Run algo > save path if goal
import tkinter as tk
import sys
from queue import Queue, LifoQueue, PriorityQueue
from enum import Enum, auto
#File input
file = [0 for _ in range(4)]
file[0] = "level1.txt"
#for testing file 5
#file.append("critical")
#Size of board m x n
boardSize = [0 for _ in range(2)]
board = []
start = []
goal = []
level = 0
time = 0
fuel = 0
#UI' componet
root = tk.Tk()
root.title("21127246")
root.geometry("500x600")
frame = tk.Frame(root, width=500, height=50, bg="white")
frame.pack(expand=True, fill=tk.BOTH)
canvas = tk.Canvas(root, width=10 * 50, height=10 * 50, bg="white")
canvas.pack()
rootAwaitFunction = []
#Flag enum
class Flag(Enum):
    notFound = auto()
    eTime = auto()
    eFuel = auto()
    validFound = auto()
#Algo enum
class Algorithm(Enum):
    bfs = 0
    dfs = auto()
    ucs = auto()
    gbfs = auto()
    a = auto()
#Create a set of valid moves to targets
def ValidMove(node):
    global boardSize, board
    result = []
    left = node["x"] - 1
    right = node["x"] + 1
    up = node["y"] - 1
    down = node["y"] + 1

    if left >= 0 and board[node["y"]][left] != -1:
        result.append({
            "x": left,
            "y": node["y"]
        })
    
    if right < boardSize[1] and board[node["y"]][right] != -1:
        result.append({
            "x": right,
            "y": node["y"]
        })
    
    if up >= 0 and board[up][node["x"]] != -1:
        result.append({
            "x": node["x"],
            "y": up
        })

    if down < boardSize[0] and board[down][node["x"]] != -1:
        result.append({
            "x": node["x"],
            "y": down
        })

    return result

#Path Backtracking
def Path(s = 0, g = 0, edges = []):
    i = g 
    path = []
    while i != s:
        path.append(i)
        i = edges[i["y"]][i["x"]]
    path.append(s)

    return path    
#Breadth first search
def BFS(pairStartGoal):
    #variables
    global board, start, goal, boardSize
    s = start[pairStartGoal]
    g = goal[pairStartGoal]
    visited = []
    queue = Queue()
    edges = [[0 for _ in range(boardSize[1])] for _ in range(boardSize[0])]

    queue.put(s)

    while not queue.empty() and g not in visited:
        current = queue.get()
        # Process the current vertex
        # check adjecents of current vertex
        neightbor = ValidMove(current)
        for i in neightbor:
            if i == g:
                visited.append(g)
                edges[g["y"]][g["x"]] = current
                break
            elif i not in visited:
                visited.append(i)
                queue.put(i)
                #Save edges
                edges[i["y"]][i["x"]] = current

    if g not in visited:
        return Flag.notFound, []

    return Flag.validFound, Path(s, g, edges);
#Deep first search
def DFS(pairStartGoal):
    #variables
    global board, start, goal, boardSize
    s = start[pairStartGoal]
    g = goal[pairStartGoal]
    expanded = []
    found = False
    stack = LifoQueue()
    edges = [[0 for _ in range(boardSize[1])] for _ in range(boardSize[0])]

    stack.put(s)

    while not stack.empty() and found == False:
        current = stack.get()
        # Process the current vertex
        # check adjecents of current vertex
        neightbor = ValidMove(current)
        for i in neightbor:
            if i == g:
                found = True
                edges[g["y"]][g["x"]] = current
                break
            elif i not in expanded:
                stack.put(i)
                #Save edges
                edges[i["y"]][i["x"]] = current
        expanded.append(current)

    if found == False:
        return Flag.notFound, []

    return Flag.validFound, Path(s, g, edges)

def C(s, n, prePath):
    if s == n:
        return 0
    
    return prePath + 1
#Uniform cost search
def UCS(pairStartGoal):
    #variables
    global board, start, goal, boardSize
    s = start[pairStartGoal]
    g = goal[pairStartGoal]
    expanded = []
    queue = PriorityQueue()
    edges = [[0 for _ in range(boardSize[1])] for _ in range(boardSize[0])]
    c = [[sys.maxsize for _ in range(boardSize[1])] for _ in range(boardSize[0])]
    c[s["y"]][s["x"]] = 0
    queue.put((c[s["y"]][s["x"]], s["x"], s["y"]))
    
    while not queue.empty() and g not in expanded:
        #Get first
        (currentC, currentX, currentY) = queue.get()
        current = {
            "x": currentX,
            "y": currentY
        }
        if current in expanded:
            continue
        # Process the current vertex
        # check adjecents of current vertex
        neightbor = ValidMove(current)
        for i in neightbor:
            #Not expand yet
            if i not in expanded:
                # If goal node i have path from current smaller than it having, update it, sort the priority Queue again
                if C(current, i, currentC) < c[i["y"]][i["x"]]:
                    c[i["y"]][i["x"]] = C(current, i, currentC)
                    edges[i["y"]][i["x"]] = current
                    queue.put((c[i["y"]][i["x"]], i["x"], i["y"]))
        expanded.append(current)

    if g not in expanded:
        return Flag.notFound, []

    return Flag.validFound, Path(s, g, edges)
#Heuristic function
def H(g, n):
    return (g["x"] - n["x"]) ** 2 + (g["y"] - n["y"]) ** 2
#Greedy best first search
def GBFS(pairStartGoal):
    #variables
    global board, start, goal, boardSize
    s = start[pairStartGoal]
    g = goal[pairStartGoal]
    visited = []
    queue = PriorityQueue()
    edges = [[0 for _ in range(boardSize[1])] for _ in range(boardSize[0])]

    queue.put((H(g, s), s["x"], s["y"]))

    while not queue.empty() and g not in visited:
        (currentH, currentX, currentY) = queue.get()
        current = {
            "x": currentX,
            "y": currentY
        }
        # Process the current vertex
        # check adjecents of current vertex
        neightbor = ValidMove(current)
        for i in neightbor:
            if i == g:
                visited.append(g)
                edges[g["y"]][g["x"]] = current
                break
            elif i not in visited:
                visited.append(i)
                queue.put((H(g, i), i["x"], i["y"]))
                #Save edges
                edges[i["y"]][i["x"]] = current

    if g not in visited:
        return Flag.notFound, []

    return Flag.validFound, Path(s, g, edges)

#A* search
def A(pairStartGoal):
    #variables
    global board, start, goal, boardSize
    s = start[pairStartGoal]
    g = goal[pairStartGoal]
    expanded = []
    queue = PriorityQueue()
    edges = [[0 for _ in range(boardSize[1])] for _ in range(boardSize[0])]
    c = [[sys.maxsize for _ in range(boardSize[1])] for _ in range(boardSize[0])]
    c[s["y"]][s["x"]] = 0
    queue.put((c[s["y"]][s["x"]] + H(g, s), s["x"], s["y"]))
    
    while not queue.empty() and g not in expanded:
        #Get first
        (currentF, currentX, currentY) = queue.get()
        current = {
            "x": currentX,
            "y": currentY
        }
        if current in expanded:
            continue
        # Process the current vertex
        # check adjecents of current vertex
        neightbor = ValidMove(current)
        for i in neightbor:
            #Not expand yet
            if i not in expanded:
                # If goal node i have path from current smaller than it having, update it, sort the priority Queue again
                if C(current, i, c[current["y"]][current["x"]]) < c[i["y"]][i["x"]]:
                    c[i["y"]][i["x"]] = C(current, i, c[current["y"]][current["x"]])
                    edges[i["y"]][i["x"]] = current
                    queue.put((c[i["y"]][i["x"]] + H(g, i), i["x"], i["y"]))
        expanded.append(current)

    if g not in expanded:
        return Flag.notFound, []

    return Flag.validFound, Path(s, g, edges)
#Check a string is number/ float
def Is_number(n):
    try:
        float(n)  # Type-casting the string to float
        return True
    except ValueError:
        return False
#Check if goal, start or special nodes overlap
def ContainOverlap():
    global start, goal
    mark = []
    for i in start:
        if i in mark:
            return True
        else:
            mark.append(i)
    
    for i in goal:
        if i in mark:
            return True
        else:
            mark.append(i)

    return False

def BoardInit():
    return [[0 for _ in range(boardSize[1])] for _ in range(boardSize[0])]

def ReadFile(mode):
    global boardSize, board, start, goal
    filename = file[mode]
    s = -1
    with open(filename, "r") as f:
        for line in f:
            values = line.strip().split()  # Remove leading/trailing spaces and split
            #First line read
            if s == -1:
                #Not valid first line
                if len(values) < 2:
                    return False
                for i in values:
                    if Is_number(i) == False:
                        return False
                #row
                boardSize[0] = int(values[0])
                #col
                boardSize[1] = int(values[1])
                #set empty board
                board = BoardInit()
            else:
                #Not full information of next line
                if len(values) < boardSize[1]:
                    return False
                #Set element in the row of board (wall or not)
                for i in range(boardSize[1]):
                    #if the value is a wall, mark it
                    if Is_number(values[i]) and int(values[i]) == -1:
                        board[s][i] = -1
                    #Get start(s) index
                    elif values[i] == "S":
                        start.append({
                            "x": i,
                            "y": s
                        })
                    #Get goal(s) index
                    elif values[i] == "G":
                        goal.append({
                            "x": i,
                            "y": s
                        })
            # Process the 'values' list as needed
            s += 1
    
    if s < boardSize[0] or ContainOverlap() or len(start) != len(goal) or (len(start) > 1 and level != 3):
        return False

    return True

def Create_grid(canvas, rows, cols, cell_size):
    for row in range(rows):
        for col in range(cols):
            x1 = col * cell_size
            y1 = row * cell_size
            x2 = x1 + cell_size
            y2 = y1 + cell_size
            color = 'white' if board[row][col] == 0 else 'black'
            canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline='black')

def Mark_position(canvas, row, col, cell_size, text, color, outline):
    x1 = col * cell_size
    y1 = row * cell_size
    x2 = x1 + cell_size
    y2 = y1 + cell_size
    canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline=outline)
    canvas.create_text((x1 + x2) // 2, (y1 + y2) // 2, text=text, fill='black')

def RunTimeBack(frame):
    global canvas, rootAwaitFunction, root

    for i in rootAwaitFunction:
        root.after_cancel(i)
    canvas, frame = AlgorithmsSelectState(canvas, frame)

def RunTimeInformation(frame):
    for child in frame.winfo_children():
        child.destroy()
    backButton = tk.Button(frame, text="< Back", font=("Helvetica", 20), command=lambda:RunTimeBack(frame))
    backButton.pack(side="left")
    process_label = tk.Label(frame, text="Process...", font=("Helvetica", 20), bg='white')
    process_label.pack(side="left")
    time_label = tk.Label(frame, text="Time", font=("Helvetica", 20), bg='white', height=1)
    time_label.pack(side="right")
   

    return process_label, frame, time_label

def BoardVisualizer(canvas, cell_size):
    global board, boardSize, start, goal
    canvas.delete("all")
    rows = boardSize[0]
    cols = boardSize[1]

    Create_grid(canvas, rows, cols, cell_size)
    for i in start:
        Mark_position(canvas, i["y"], i["x"], cell_size, 'S', '#d5e8d4', 'green')
        
    for i in goal:
        Mark_position(canvas, i["y"], i["x"], cell_size, 'G', '#f8cecc', 'red')
    
    return canvas

def StepVisualizer(time_label, root, canvas, cell_size, step, path, time = 0, timeStep = 1):
    global rootAwaitFunction
    if step < len(path) - 1:
        time_label.config(text="Time: "+ str(time) + "s")
        if step >= 0:
            rootAwaitFunction.pop(0)
            i = path[len(path) - step - 1]
            #clear current
            Mark_position(canvas, i["y"], i["x"], cell_size, '', 'white', 'black')
            #Display to view
        i = path[len(path) - step - 2]
        #go to next
        Mark_position(canvas, i["y"], i["x"], cell_size, 'S', '#d5e8d4', 'green')
        # Define a lambda function that calls your main function with arguments
        call_with_args = lambda: StepVisualizer(time_label, root, canvas, cell_size, step + 1, path, time + timeStep)
        rootAwaitFunction.append(root.after(1000, call_with_args))
#Select algorithm in level 1
def AlgorithmPicker(algorithm):
    if algorithm == Algorithm.bfs:
        return BFS(0)
    elif algorithm == Algorithm.dfs:
        return DFS(0)
    elif algorithm == Algorithm.ucs:
        return UCS(0)
    elif algorithm == Algorithm.gbfs:
        return GBFS(0)
    return A(0)

def Level1(canvas, frame):
    global level
    level = 1
    return AlgorithmsSelectState(canvas, frame);


def RuntimeState(root, canvas, frame, algorithm):
    global Flag, rootAwaitFunction, level
    label, frame, time_label = RunTimeInformation(frame)
    cell_size = 500 / boardSize[0]
    if boardSize[0] < boardSize[1]:
        cell_size = 500 / boardSize[1]
    canvas = BoardVisualizer(canvas, cell_size)

    flag, path = AlgorithmPicker(algorithm)
    if flag == Flag.validFound:
        label.config(text='Found!', fg='Green')
        step = -1
        # Define a lambda function that calls your main function with arguments
        StepVisualizer(time_label, root, canvas, cell_size, step, path)
    elif flag == Flag.notFound:
        label.config(text='Not Found!', fg='Red')

    return canvas, frame

def AlgorithmsSelectLabel(frame):
    for child in frame.winfo_children():
        child.destroy()
    label = tk.Label(frame, text="Select an algorithm", font=("Helvetica", 20), bg='white')
    label.pack()

    return frame

def AlgorithmsSelectButtons(canvas):
    global root, frame
    canvas.delete("all")
    y = 0
    tag = {
            Algorithm.bfs: "BFS", 
            Algorithm.dfs: "DFS", 
            Algorithm.ucs: "UCS", 
            Algorithm.gbfs: "GBFS", 
            Algorithm.a: "A*"
        }

    for e, i in Algorithm.__members__.items():
        topMargin = 20
        if i == Algorithm.bfs:
            topMargin = 30
        y += topMargin
        button = tk.Button(canvas, 
                       text=tag[i], 
                       command=lambda i=i:RuntimeState(root, canvas, frame, i), 
                       anchor=tk.W,
                       font=('Arial', 20),
                       compound="c",
                       width=5,
                       height=1,
                       justify="center")
        canvas.create_window(210, y, anchor=tk.NW, window=button)
        y += 60
    
    return canvas

def AlgorithmsSelectState(canvas, frame):
    frame = AlgorithmsSelectLabel(frame)
    canvas = AlgorithmsSelectButtons(canvas)
    return canvas, frame

def LevelsSelectLabel(frame):
    for child in frame.winfo_children():
        child.destroy()
    label = tk.Label(frame, text="Select a level", font=("Helvetica", 20), bg='white')
    label.pack()

    return frame

def LevelsSelectButtons(canvas):
    global root, frame
    canvas.delete("all")
    y = 0
    levelList = [lambda: Level1(canvas, frame)]

    for i in range(1):
        topMargin = 20
        if i == Algorithm.bfs:
            topMargin = 30
        y += topMargin
        button = tk.Button(canvas, 
                       text=str(i + 1), 
                       command=lambda i=levelList[i]:RuntimeState(root, canvas, frame, i), 
                       anchor=tk.W,
                       font=('Arial', 20),
                       compound="c",
                       width=5,
                       height=1,
                       justify="center")
        canvas.create_window(210, y, anchor=tk.NW, window=button)
        y += 60
    
    return canvas

def main():
    global root, canvas, frame
    if ReadFile(0) == False:
        # Create a text widget
        # Create a label with red text color and large font size
        canvas.create_text(100, 235, anchor="nw", text="Not valid input file", fill='red', font=("Arial", 30))
    else:
        canvas, frame = Level1(canvas, frame)

    root.mainloop()

if __name__ == "__main__":
     main()
