"""
research.py - extra experiments for the Research section (COS30019 A2A)

Compares all six search methods on square grid problems of growing size
(origin = bottom-left corner, destination = top-right corner) and prints
tables of nodes created, moves, path cost, time and peak memory.


"""

import random
import time
import tracemalloc

import search

METHODS = ("DFS", "BFS", "GBFS", "AS", "CUS1", "CUS2")
SIZES = (3, 4, 5, 6)                     # grid side lengths to test


def grid_problem(side, varied_costs=False, seed=1):
    """side x side grid, two-way roads between neighbours.
    Unit costs, or random whole-number costs 1-9 (never below the road's
    straight-line length of 1, so A* stays optimal)."""
    rng = random.Random(seed)
    coordinates = {r * side + c + 1: (c, r) for r in range(side) for c in range(side)}
    edges = {}
    for r in range(side):
        for c in range(side):
            a = r * side + c + 1
            neighbours = []
            if c < side - 1:
                neighbours.append(a + 1)
            if r < side - 1:
                neighbours.append(a + side)
            for b in neighbours:
                cost = rng.randint(1, 9) if varied_costs else 1
                edges[(a, b)] = cost
                edges[(b, a)] = cost
    return search.Problem(coordinates, edges, 1, [side * side])


def run_once(method, problem):
    """Returns (nodes_created, moves, cost, seconds, peak_kb)."""
    function = search.SEARCH_FUNCTIONS[method]

    start = time.perf_counter()
    goal, created = function(problem)
    seconds = time.perf_counter() - start

    tracemalloc.start()                  # second run only to measure memory
    function(problem)
    peak_kb = tracemalloc.get_traced_memory()[1] / 1024
    tracemalloc.stop()

    return created, goal.depth, goal.cost, seconds, peak_kb


def print_table(title, cases, column, fmt):
    print("\n%s" % title)
    print("| Grid | " + " | ".join(METHODS) + " |")
    print("| --- |" + " --- |" * len(METHODS))
    for label, results in cases:
        cells = [fmt % results[m][column] for m in METHODS]
        print("| %s | %s |" % (label, " | ".join(cells)))


def main():
    for varied in (False, True):
        kind = "random road costs 1-9" if varied else "unit road costs"
        cases = []
        for side in SIZES:
            problem = grid_problem(side, varied)
            results = {m: run_once(m, problem) for m in METHODS}
            cases.append(("%dx%d" % (side, side), results))
        print("\n================ %s ================" % kind)
        print_table("Nodes created", cases, 0, "%d")
        print_table("Moves in the path found", cases, 1, "%d")
        print_table("Cost of the path found", cases, 2, "%d")
        print_table("Time (seconds)", cases, 3, "%.3f")
        print_table("Peak memory (KB)", cases, 4, "%.0f")


if __name__ == "__main__":
    main()