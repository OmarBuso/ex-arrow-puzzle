# Arrow puzzle

Recreation of the arrow puzzle from exponential idle made with Python and Pygame
Make all circles turn black by tapping the tiles
Expect some bugs (screen sizing, solver not working, etc.)

## Features

• Hexagonal grid with adjustable size and number of states (default: 4 cells per side and 6 states)

• Random solvable board generation

• Solver mode to manually edit boards and show solution hints

• "Generate Random Board" | "Reset board" button: Generates puzzle (board) or animates the solution

• Built in NumPy solver (mod 6) for fast solutions

## Solver

The solver works over mod 6 arithmetic using Gaussian elimination

Ax ≡ b (mod 6)

Hint numbers in yellow show which circles should be pressed and how many times

## License

MIT License, free to use, modify, and share
