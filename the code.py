import pygame
import math
import random
import time
try:
    import numpy as np
except ImportError:
    import os
    os.system("pip install numpy")
from itertools import product

# Settings
widthX, heightY = 1080, 2100 # Might be different for every device
hexSize = 100 # Hexagon size
gridRadius = 3 # Number of hexagons + 1 in every side
totalStates = 2

# Colors
colorBackground = (50, 50, 50)
colorLine = (120, 120, 120)
colorWhite = (255, 255, 255)
colorButton = (60, 60, 60)
colorActiveButton = (90, 90, 90)
colorHint = (255, 255, 0)

colorState = []
if totalStates == 2:
	colorState = [
		(30, 30, 30), # 1
		(105, 105, 105) # 2
	]
if totalStates == 6:
	colorState = [
		(30, 30, 30), # 1
		(45, 45, 45), # 2
		(60, 60, 60), # 3
		(75, 75, 75), # 4
		(90, 90, 90), # 5
		(105, 105, 105) # 6
	]
# If for some reason the player chooses to use a number of states other than 2 and 6, this thing will make new colors
for c in range(totalStates):
		col = c * 255//totalStates
		colorState.append((col, col, col))

pygame.init()
screen = pygame.display.set_mode((widthX, heightY))
pygame.display.set_caption("Arrow Puzzle")
clock = pygame.time.Clock()
font = pygame.font.SysFont("consolas", 60) # Looks a bit like the font the game uses

## If you dont know what you're doing, its recommended to not move anything past this point

# Directions for the Neighbor cells
directions = [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]

def hex_to_pixel(q, r, size=hexSize): # I mean, the board is tecnically a hexagon, right?
    x = size * (3/2 * q)
    y = size * (math.sqrt(3) * (r + q/2))
    return x, y

def pixel_to_hex(x, y, size=hexSize):
    q = (2/3 * x) / size
    r = (-1/3 * x + math.sqrt(3)/3 * y) / size
    return hex_round(q, r)

def hex_round(q, r):
    s = -q - r
    rq = round(q)
    rr = round(r)
    rs = round(s)

    dq = abs(rq - q)
    dr = abs(rr - r)
    ds = abs(rs - s)

    if dq > dr and dq > ds:
        rq = -rr - rs
    elif dr > ds:
        rr = -rq - rs
    return rq, rr

# Generate hexagonal grid
hexGrid = {}
for q in range(-gridRadius, gridRadius+1):
    for r in range(-gridRadius, gridRadius+1):
        if abs(q+r) <= gridRadius:
            hexGrid[(q,r)] = 0

def change_neighbors(q, r):
    if (q,r) in hexGrid:
        hexGrid[(q, r)] = (hexGrid[(q, r)] + 1) % totalStates # 1 -> 2 -> ... -> n -> 1
    for dq, dr in directions:
        neighbor = (q+dq, r+dr)
        if neighbor in hexGrid:
            hexGrid[neighbor] = (hexGrid[neighbor] + 1) % totalStates

def random_board():
    # Resets board to avoid giving a solved one
    for k in hexGrid:
        hexGrid[k] = 0
    moves = random.randint(5*totalStates^2,10*totalStates^2)
    # Applies random moves
    for _ in range(moves):
        q, r = random.choice(list(hexGrid.keys()))
        change_neighbors(q, r)

keys = list(hexGrid.keys())

def matrix(keys, hexGrid, directions):
    # Finds x (clicks in the grid) to solve Ax = b (mod N), where A is the matrix and b is the entire grid
    n = len(keys)
    A = np.zeros((n,n), dtype=np.uint8)
    for i, (q, r) in enumerate(keys):
        A[i,i] = 1
        for dq, dr in directions:
            v = (q + dq, r + dr)
            if v in hexGrid:
                j = keys.index(v)
                A[i,j] = 1
    return A % 2

# Of course there has to be linear algebra...
def solution_boardN(A, b, N):
    A = A.copy() % N
    b = b.copy() % N
    n, m = A.shape
    aug = np.concatenate([A, b.reshape(-1, 1)], axis=1)
    rank = 0
    pivots = []

    # Modular inverse function
    def mod_inv(a, N):
        a = a % N
        for x in range(1, N):
            if (a * x) % N == 1:
                return x
        return None  # No inverse (if not coprime)

    for col in range(m):
        pivot_row = None
        for r in range(rank, n):
            if aug[r, col] % N != 0 and mod_inv(aug[r, col], N) is not None:
                pivot_row = r
                break
        if pivot_row is None:
            continue

        aug[[rank, pivot_row]] = aug[[pivot_row, rank]]
        pivots.append(col)

        inv = mod_inv(aug[rank, col], N)
        aug[rank, :] = (aug[rank, :] * inv) % N

        for r in range(n):
            if r != rank and aug[r, col] != 0:
                factor = aug[r, col]
                aug[r, :] = (aug[r, :] - factor * aug[rank, :]) % N

        rank += 1
        if rank == n:
            break

    # Check for inconsistency
    for r in range(n):
        if np.all(aug[r, :-1] % N == 0) and aug[r, -1] % N != 0:
            return False, None, None

    # Extract solution (particular)
    x = np.zeros(m, dtype=np.int64)
    for i, c in enumerate(pivots):
        x[c] = aug[i, -1] % N

    # Nullspace basis
    freeCols = [c for c in range(m) if c not in pivots]
    nullspace = []
    for f in freeCols:
        z = np.zeros(m, dtype=np.int64)
        z[f] = 1
        for i, c in enumerate(pivots):
            z[c] = (-aug[i, f]) % N
        nullspace.append(z)

    return True, x, nullspace

def hamming_weight(v):
	return int(v.sum())

def search_solutions(xPart, nullspace, limit = 20):
	k = len(nullspace)
	if k == 0 or k > limit:
		return xPart
	best = xPart.copy()
	bestWeight = hamming_weight(xPart)
	for bits in product([0,1], repeat=k):
		u = xPart.copy()
		for i, bit in enumerate(bits):
			if bit:
				u ^= nullspace[i]
		weight = hamming_weight(u)
		if weight < bestWeight:
			best, bestWeight = u, weight
	return best

def solve_board(keys, hexGrid, directions, limitSearch=50):
    A = matrix(keys, hexGrid, directions)
    b = np.array([hexGrid[k] % totalStates for k in keys], dtype=np.int64)
    solvable, xPart, nullspace = solution_boardN(A, b, totalStates)
    if not solvable:
        return None, "No solutions"
    x = xPart % totalStates

    # Return dictionary with presses
    pressDict = {keys[i]: int(x[i]) for i in range(len(keys)) if x[i] != 0}
    totalPresses = sum(pressDict.values())
    return pressDict, f"Steps: {totalPresses}"

def expand_solution(solutionDict):
    sequence = []
    for (q, r), presses in solutionDict.items():
        # Invert the move mod totalStates
        inverse_presses = (totalStates - presses) % totalStates
        for _ in range(inverse_presses):
            sequence.append((q, r))
    return sequence

def draw_circle(x, y, size, state, hint=None):
  	color = colorState[state]
    
  	pygame.draw.circle(screen, color, (x, y), size)
  	pygame.draw.circle(screen, colorLine, (x, y), size, 2)
  	fontCell = pygame.font.SysFont("consolas", 50, bold=True)
  	text = fontCell.render(str(state + 1), True, colorWhite)
  	rect = text.get_rect(center=(x, y))
  	screen.blit(text, rect)
  	
  	# If there is a hint, show above
  	if hint is not None:
  	     fontHint = pygame.font.SysFont("consolas", 60, bold=True)
  	     numberHint = fontHint.render(str(totalStates - hint), True, colorHint)
  	     screen.blit(numberHint, numberHint.get_rect(center=(x + 30, y - 30)))

def solved_board():
    return all(v == 0 for v in hexGrid.values())

def format_time(seconds):
	if seconds is None:
		return "00:00.000"
	if seconds < 0:
	  seconds = 0.0
	minutes = int(seconds // 60)
	sec = seconds - minutes * 60
	return f"{minutes:02d}:{sec:06.3f}"

buttonGame = pygame.Rect(widthX/2-500, heightY - 100, 1000, 100) 
buttonSolver = pygame.Rect(widthX/2-500, heightY - 250, 1000, 100)

# Game states
activeGame = False
activeSolver = False

# Reset Animation states
solvingAnimation = False
solutionSequence = []
animationTimer = 0
animationDelay = 200//totalStates  # Milliseconds between presses

# Times
startingTime = None
totalTime = 0.0
bestTime = None

# Main loop
running = True
while running:
    screen.fill(colorBackground)
    
    # Calculate hints if solver mode is on
    solution, message = solve_board(keys, hexGrid, directions)
    if activeSolver:
    	textSolvable = font.render(message, True, colorWhite)
    	screen.blit(textSolvable, (widthX//2 - 220, heightY - 400)) if message == "No solutions" else screen.blit(textSolvable, (widthX//2 - 150, heightY - 400))
    	
    if activeGame and startingTime is not None:
        totalTime = time.time() - startingTime
    
    # Draw times
    textTime = font.render(f"Time: {format_time(totalTime)}", True, colorWhite)
    screen.blit(textTime, (widthX//2 - 400, 140))
    textBest = font.render(f"Best Time: {format_time(bestTime)}", True, colorWhite)
    screen.blit(textBest, (widthX//2 - 400, 210))
    
    # Draw circles
    for (q, r) in keys:
        x, y = hex_to_pixel(q, r, hexSize)
        x += widthX // 2
        y += heightY // 2 - 12
        state = hexGrid[(q, r)]
        hintValue = None
        if solvingAnimation and (q, r) in solution:
        	hintValue = solution[(q, r)]
        elif activeSolver and solution is not None and (q, r) in solution:
        	hintValue = solution[(q, r)]
        draw_circle(x, y, hexSize - 12, state, hintValue)

    # Draw buttons
    mouse = pygame.mouse.get_pos()
    hover1 = buttonGame.collidepoint(mouse)
    hover2 = buttonSolver.collidepoint(mouse)
    buttonColor1 = colorActiveButton if hover1 else colorButton
    buttonColor2 = colorActiveButton if hover2 else colorButton
    pygame.draw.rect(screen, buttonColor1, buttonGame, border_radius=12)
    pygame.draw.rect(screen, buttonColor2, buttonSolver, border_radius=12)
    
    text1 = "Generate random board" if not activeGame or activeSolver else "Reset Board"
    screen.blit(font.render(text1, True, colorWhite), (buttonGame.x + 20, buttonGame.y + 10))
    
    text2 = "Switch to Solver mode" if not activeSolver else "Switch to Play mode"
    screen.blit(font.render(text2, True, colorWhite), (buttonSolver.x + 20, buttonSolver.y + 10))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            raise SystemExit
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = event.pos
            
            # Solver button
            if buttonSolver.collidepoint(mx, my):
            	activeSolver = not activeSolver
            	if not solved_board():
            		activeGame = not activeGame
            	if activeSolver:
            		startingTime = None
            		totalTime = 0.0
            	continue
            	
            # Game button
            if buttonGame.collidepoint(mx, my):
                if activeGame:
                	if solution is not None:
                		solutionSequence = expand_solution(solution)
                		solvingAnimation = True
                		animationTimer = pygame.time.get_ticks()
                		activeGame = False
                		startingTime = None
                		totalTime = 0.0
                	else:
                		for k in hexGrid: hexGrid[k] = 0 # Resets direclty if unsolvable
                		activeGame = False
                		startingTime = None
                		totalTime = 0.0
                else:
                	if activeSolver:
                		random_board()
                		activeGame = False
                		startingTime = None
                		totalTime = 0.0
                	if not activeSolver:
                		random_board()
                		activeGame = True
                		startingTime = None
                		totalTime = 0.0
            
            q, r = pixel_to_hex(mx - widthX//2, my - heightY//2 + 50)
            if (q, r) in hexGrid:
            	if activeSolver:
            		hexGrid[(q, r)] = (hexGrid[(q, r)] + 1) % totalStates
            	elif activeGame:
            		if startingTime is None:
            			startingTime = time.time()
            		change_neighbors(q, r)
            if activeGame and solved_board():
            	finalTime = time.time() - startingTime if startingTime is not None else 0.0
            	if bestTime is None or finalTime < bestTime:
            		bestTime = finalTime
            		startingTime = None
            	activeGame = False

		# Creates resetting animation
    if solvingAnimation and len(solutionSequence) > 0:
    	currentTime = pygame.time.get_ticks()
    	if currentTime - animationTimer >= animationDelay:
    	 q, r = solutionSequence.pop(0)
    	 change_neighbors(q, r)
    	 animationTimer = currentTime
    	 
    	# Stops animation when finished
    	if len(solutionSequence) == 0:
    		solvingAnimation = False
    pygame.display.flip()
    clock.tick(30)

pygame.quit()

# Just if I need to say this:
# Variables, lists [] and dictionaries {} have names with something like "oneTwo"
# Functions have names like "function_name()"
# Basically, one uses underscores to separate words, while the other one uses a capital letter
