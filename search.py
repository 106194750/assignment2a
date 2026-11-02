"""
COS30019 Introduction to AI - Assignment 2, Part A

usage:
    python search.py <filename> <method>

methods:
    DFS   depth-first search                      (uninformed)
    BFS   breadth-first search                    (uninformed)
    GBFS  greedy best-first search                (informed)
    AS    A* search                               (informed)
    CUS1  iterative-deepening depth-first search  (uninformed, custom 1)
    CUS2  IDA* on number of moves                 (informed, custom 2)

Output when goal is reached:
    filename method
    goal number_of_nodes
    path

Output when no goal can be reached:
    filename method
    No goal is reachable; number_of_nodes
"""

import heapq
import itertools
import math
import re
import sys
from collections import deque


PATH_SEPARATOR = " -> "

# global variable to store the names of methods
METHODS = ("DFS", "BFS", "GBFS", "AS", "CUS1", "CUS2")


# --------------------------------------------------------------------------
# Problem definition and file parser
# --------------------------------------------------------------------------

class Problem:
    """problem gets read from file"""

    def __init__(self, coordinates, edges, origin, destinations):
        self.coordinates = coordinates      # {node: (x, y)}
        self.origin = origin                # start node
        self.destinations = destinations    # list of goal nodes

        # adjacency[a] = [(b, cost), ...] sorted by b so that children are
        # always generated in ascending node order (tie-break rule).
        self.adjacency = {node: [] for node in coordinates}
        for (a, b), cost in edges.items():
            self.adjacency[a].append((b, cost))
        for node in self.adjacency:
            self.adjacency[node].sort()

        self._goals = set(destinations)

    def is_goal(self, node):
        return node in self._goals

    def distance(self, a, b):
        """distance between two nodes."""
        (x1, y1), (x2, y2) = self.coordinates[a], self.coordinates[b]
        return math.hypot(x1 - x2, y1 - y2)

    def distance_to_nearest_goal(self, node):
        """heuristic for GBFS and A*, distance to the closest destination."""
        return min(self.distance(node, goal) for goal in self.destinations)

    def longest_edge_length(self):
        """longest straight-line length of any single edge (for CUS2)."""
        longest = 0.0
        for a, neighbours in self.adjacency.items():
            for b, _ in neighbours:
                longest = max(longest, self.distance(a, b))
        return longest


class FileFormatError(Exception):
    """for when the problem file isnt in the right format"""


_NODE_LINE = re.compile(r"^(\d+)\s*:\s*\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)$")
_EDGE_LINE = re.compile(r"^\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*:\s*(-?[\d.]+)$")
_SECTIONS = ("nodes", "edges", "origin", "destinations")


def read_problem(filename):
    """parse a problem file. handles blank lines, extra spaces and line breaks..."""
    coordinates, edges = {}, {}
    origin, destinations = None, []
    section = None

    with open(filename, "r") as handle:
        lines = handle.read().splitlines()

    for number, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line:
            continue

        # section keyword like "Nodes:" or "Origin: 2" (value on same line).
        keyword, _, rest = line.partition(":")
        if keyword.strip().lower() in _SECTIONS:
            section = keyword.strip().lower()
            line = rest.strip()
            if not line:
                continue

        try:
            if section == "nodes":
                match = _NODE_LINE.match(line)
                node, x, y = match.groups()
                coordinates[int(node)] = (float(x), float(y))
            elif section == "edges":
                match = _EDGE_LINE.match(line)
                a, b, cost = match.groups()
                edges[(int(a), int(b))] = float(cost)
            elif section == "origin":
                origin = int(line)
            elif section == "destinations":
                destinations = [int(t) for t in re.split(r"[;,]", line) if t.strip()]
            else:
                raise FileFormatError("text before the first section keyword")
        except (AttributeError, ValueError):
            raise FileFormatError("line %d is not in the expected format: %r"
                                  % (number, raw))

    # sanity checks so later code can assume a consistent problem.
    if origin is None or not destinations or not coordinates:
        raise FileFormatError("missing Nodes, Origin or Destinations section")
    for node in [origin] + destinations + [n for e in edges for n in e]:
        if node not in coordinates:
            raise FileFormatError("node %d is used but has no coordinates" % node)

    return Problem(coordinates, edges, origin, destinations)


# --------------------------------------------------------------------------
# Search tree
# --------------------------------------------------------------------------

class TreeNode:
    """a node of the search tree (not the same thing as node on graph)"""
    __slots__ = ("state", "parent", "cost", "depth")

    def __init__(self, state, parent=None, cost=0.0):
        self.state = state                                  # graph node number
        self.parent = parent                                # previous tree node
        self.cost = cost                                    # g(n): path cost so far
        self.depth = 0 if parent is None else parent.depth + 1   # number of moves

    def path(self):
        """graph nodes from the origin to this node"""
        states, node = [], self
        while node is not None:
            states.append(node.state)
            node = node.parent
        return states[::-1]


def expand(problem, node):
    """children of a tree node in ascending node order, skipping any graph
    node that is already on the path from the root..."""
    on_path = set(node.path())
    return [TreeNode(nxt, node, node.cost + cost)
            for nxt, cost in problem.adjacency[node.state]
            if nxt not in on_path]


# --------------------------------------------------------------------------
# search methods. each one returns (goal_tree_node or None, nodes_created).
# --------------------------------------------------------------------------

def depth_first_search(problem):
    """DFS always expands the deepest node. the lowest-numbered child first."""
    root = TreeNode(problem.origin)
    frontier = [root]                       # used as a stack
    created = 1
    while frontier:
        node = frontier.pop()
        if problem.is_goal(node.state):
            return node, created
        children = expand(problem, node)
        created += len(children)
        # in reverse so the smallest-numbered child is popped first.
        frontier.extend(reversed(children))
    return None, created


def breadth_first_search(problem):
    """BFS: expand nodes level by level, oldest node first."""
    root = TreeNode(problem.origin)
    frontier = deque([root])                # used as a queue 
    created = 1
    while frontier:
        node = frontier.popleft()
        if problem.is_goal(node.state):
            return node, created
        children = expand(problem, node)
        created += len(children)
        frontier.extend(children)
    return None, created


def best_first_search(problem, evaluate):
    """generic 'best-first' search that we use for both GBFS and A*. `evaluate(tree_node)` gives the value to
    minimise. breaks ties.
    see https://www.geeksforgeeks.org/dsa/best-first-search-informed-search/ for more..."""
    order = itertools.count()               # creation order for the last tie-break

    def entry(tree_node):
        return (round(evaluate(tree_node), 9), tree_node.state, next(order), tree_node)

    root = TreeNode(problem.origin)
    frontier = [entry(root)]                # binary heap
    created = 1
    while frontier:
        node = heapq.heappop(frontier)[-1]
        if problem.is_goal(node.state):
            return node, created
        children = expand(problem, node)
        created += len(children)
        for child in children:
            heapq.heappush(frontier, entry(child))
    return None, created

"""we could have avoided using lambdas here, but it makes everything simple."""

def greedy_best_first_search(problem):
    """for GBFS, f(n) = h(n), only uses the estimated distance to the goal."""
    return best_first_search(
        problem, lambda n: problem.distance_to_nearest_goal(n.state))


def a_star_search(problem):
    """for A*, f(n) = g(n) + h(n), path cost so far plus estimated distance left."""
    return best_first_search(
        problem, lambda n: n.cost + problem.distance_to_nearest_goal(n.state))


def iterative_deepening_search(problem):
    """ this is CUS1, iterative-deepening DFS (uninformed).

    runs DFS with limits on depth (0, 1, 2, ...) until a goal node is found.
    it finds the shallowest goal like BFS does.
    number_of_nodes is the total created over all iterations, because the
    program recreates the upper levels on every pass...
    search stops when a pass is never cut off by the limit, because that means
    the whole tree got explored and no goal was found."""
    created = 0
    limit = 0
    while True:
        root = TreeNode(problem.origin)
        frontier = [root]
        created += 1
        cut_off = False                     # did the depth limit hide anything
        while frontier:
            node = frontier.pop()
            if problem.is_goal(node.state):
                return node, created
            children = expand(problem, node)
            if node.depth == limit:
                cut_off = cut_off or bool(children)
                continue                    # at the limit, do not go deeper
            created += len(children)
            frontier.extend(reversed(children))
        if not cut_off:
            return None, created
        limit += 1


def ida_star_search(problem):
    """CUS2 - IDA* (iterative-deepening A*) where every move costs 1
    (informed, fewest moves).

    f(n) = moves so far + h(n), where h(n) is the straight-line distance to
    the nearest goal divided by the longest single edge in the graph, rounded
    up. No move can cover more than that distance, so at least that many moves
    are still needed: h never overestimates, which makes IDA* optimal in moves
    (path cost is ignored).

    How it works: run a depth-first search that refuses to go into any node
    whose f is above a threshold. If no goal is found, the new threshold is the
    smallest f that was refused, and the search starts again. This gives A*'s
    answer but keeps only the current path in memory, like DFS.

    number_of_nodes is the total created over ALL iterations. A child counts as
    created even if it is refused, because its f had to be calculated first."""
    longest = problem.longest_edge_length()

    def estimated_moves_left(state):
        if longest == 0:
            return 0
        return math.ceil(problem.distance_to_nearest_goal(state) / longest - 1e-9)

    def f_value(tree_node):
        return tree_node.depth + estimated_moves_left(tree_node.state)

    created = 0
    threshold = estimated_moves_left(problem.origin)    # f of the root (depth 0)
    while True:
        root = TreeNode(problem.origin)
        frontier = [root]
        created += 1
        smallest_refused = math.inf          # becomes the next threshold
        while frontier:
            node = frontier.pop()
            if problem.is_goal(node.state):
                return node, created
            children = expand(problem, node)
            created += len(children)
            allowed = []
            for child in children:
                value = f_value(child)
                if value <= threshold:
                    allowed.append(child)
                else:
                    smallest_refused = min(smallest_refused, value)
            # Push in reverse so the smallest-numbered child is popped first.
            frontier.extend(reversed(allowed))
        if smallest_refused == math.inf:
            return None, created             # nothing was refused: search space exhausted
        threshold = smallest_refused


SEARCH_FUNCTIONS = {
    "DFS": depth_first_search,
    "BFS": breadth_first_search,
    "GBFS": greedy_best_first_search,
    "AS": a_star_search,
    "CUS1": iterative_deepening_search,
    "CUS2": ida_star_search,
}


# --------------------------------------------------------------------------
# CLI stuff
# --------------------------------------------------------------------------

def main(argv):
    if len(argv) != 3:
        print("Usage: python search.py <filename> <method>")
        print("Methods: " + " ".join(METHODS))
        return 1

    filename, method = argv[1], argv[2].upper()
    if method not in SEARCH_FUNCTIONS:
        print("Unknown method '%s'. Choose one of: %s" % (argv[2], " ".join(METHODS)))
        return 1

    try:
        problem = read_problem(filename)
    except OSError as error:
        print("Cannot open '%s': %s" % (filename, error.strerror))
        return 1
    except FileFormatError as error:
        print("Problem file error: %s" % error)
        return 1

    goal, created = SEARCH_FUNCTIONS[method](problem)

    print("%s %s" % (filename, method))
    if goal is None:
        print("No goal is reachable; %d" % created)
    else:
        print("%d %d" % (goal.state, created))
        print(PATH_SEPARATOR.join(str(state) for state in goal.path()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))