"""
Assignment 1 - Question 2: Constraint Satisfaction Problem
Security Bot Scheduling

Author: AI Assignment Solution
Course: AI CSL7610

Problem:
- Variables: {Slot1, Slot2, Slot3, Slot4}
- Domains: Each slot -> {A, B, C}
- Constraints:
  1. No Back-to-Back: A bot cannot work two consecutive slots
  2. Maintenance Break: Bot C cannot work in Slot 4
  3. Minimum Coverage: Every bot must be used at least once

Input: read from input.txt
Output: Success/Failure, Heuristic, Inference, Constraints, Assignment, Stats, Time
"""

import time
from itertools import product

# PROBLEM DEFINITION


def read_csp_input(filename="input.txt"):
    """Read CSP configuration from input.txt."""
    bots = ['A', 'B', 'C']
    slots = ['Slot1', 'Slot2', 'Slot3', 'Slot4']
    unary_constraints = {}
    try:
        with open(filename, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        for line in lines:
            if line.lower().startswith("bots") or line.lower().startswith("domain"):
                parts = line.split(":", 1)
                if len(parts) > 1:
                    bots = [b.strip() for b in parts[1].split(",")]
            elif line.lower().startswith("slots") or line.lower().startswith("variable"):
                parts = line.split(":", 1)
                if len(parts) > 1:
                    slots = [s.strip() for s in parts[1].split(",")]
            elif line.lower().startswith("unary"):
                parts = line.split(":", 1)
                if len(parts) > 1:
                    # format: "Slot4 != C"
                    constraint = parts[1].strip()
                    tokens = constraint.split()
                    if len(tokens) >= 3 and tokens[1] == '!=':
                        slot = tokens[0]
                        bot = tokens[2]
                        if slot not in unary_constraints:
                            unary_constraints[slot] = []
                        unary_constraints[slot].append(bot)
    except FileNotFoundError:
        pass

    if not unary_constraints:
        unary_constraints = {'Slot4': ['C']}

    return bots, slots, unary_constraints



# CSP SOLVER WITH BACKTRACKING + MRV + FORWARD CHECKING


class CSPSolver:
    def __init__(self, bots, slots, unary_constraints):
        self.slots = slots
        self.bots = bots
        self.unary_constraints = unary_constraints  # {slot: [excluded bots]}

        # Initialize domains with unary constraints applied
        self.initial_domains = {}
        for slot in slots:
            domain = list(bots)
            if slot in unary_constraints:
                domain = [b for b in domain if b not in unary_constraints[slot]]
            self.initial_domains[slot] = domain

        self.assignments = 0
        self.constraints_applied = [
            "No Back-to-Back: A bot cannot work two consecutive slots",
            "Maintenance Break: Bot C cannot work in Slot 4",
            "Minimum Coverage: Every bot must be used at least once"
        ]

    def is_consistent(self, assignment, var, value):
        """Check if assigning value to var is consistent with constraints."""
        idx = self.slots.index(var)

        # No Back-to-Back constraint
        if idx > 0:
            prev_slot = self.slots[idx - 1]
            if prev_slot in assignment and assignment[prev_slot] == value:
                return False
        if idx < len(self.slots) - 1:
            next_slot = self.slots[idx + 1]
            if next_slot in assignment and assignment[next_slot] == value:
                return False

        return True

    def mrv_select(self, assignment, domains):
        """Select unassigned variable with Minimum Remaining Values."""
        unassigned = [v for v in self.slots if v not in assignment]
        return min(unassigned, key=lambda v: len(domains[v]))

    def forward_check(self, assignment, domains, var, value):
        """
        Forward Checking: After assigning value to var,
        prune inconsistent values from neighboring domains.
        Returns new domains or None if a domain becomes empty.
        """
        new_domains = {k: list(v) for k, v in domains.items()}
        idx = self.slots.index(var)

        # Prune next slot (no back-to-back)
        if idx + 1 < len(self.slots):
            next_slot = self.slots[idx + 1]
            if next_slot not in assignment:
                if value in new_domains[next_slot]:
                    new_domains[next_slot].remove(value)
                if not new_domains[next_slot]:
                    return None

        # Prune previous slot (no back-to-back)
        if idx - 1 >= 0:
            prev_slot = self.slots[idx - 1]
            if prev_slot not in assignment:
                if value in new_domains[prev_slot]:
                    new_domains[prev_slot].remove(value)
                if not new_domains[prev_slot]:
                    return None

        return new_domains

    def check_minimum_coverage(self, assignment):
        """Check if all bots are used at least once."""
        used = set(assignment.values())
        return all(b in used for b in self.bots)

    def backtrack(self, assignment, domains):
        """Backtracking search with MRV and Forward Checking."""
        if len(assignment) == len(self.slots):
            if self.check_minimum_coverage(assignment):
                return dict(assignment)
            return None

        var = self.mrv_select(assignment, domains)

        for value in domains[var]:
            self.assignments += 1
            if self.is_consistent(assignment, var, value):
                assignment[var] = value
                new_domains = self.forward_check(assignment, domains, var, value)

                if new_domains is not None:
                    result = self.backtrack(assignment, new_domains)
                    if result is not None:
                        return result

                del assignment[var]

        return None

    def solve(self):
        """Solve the CSP and return results."""
        t0 = time.time()
        result = self.backtrack({}, {k: list(v) for k, v in self.initial_domains.items()})
        elapsed = time.time() - t0

        return {
            "success": result is not None,
            "assignment": result,
            "total_assignments": self.assignments,
            "time": elapsed,
            "heuristic": "MRV (Minimum Remaining Values)",
            "inference": "Forward Checking",
            "constraints": self.constraints_applied
        }


# AC-3 (Arc Consistency)


def ac3_check(bots, slots, unary_constraints):
    """
    Demonstrate AC-3 arc consistency for the No Back-to-Back constraint.
    Returns whether the CSP is arc-consistent and the reduced domains.
    """
    # Initialize domains
    domains = {}
    for slot in slots:
        domain = list(bots)
        if slot in unary_constraints:
            domain = [b for b in domain if b not in unary_constraints[slot]]
        domains[slot] = domain

    # Build arcs (pairs of consecutive slots)
    arcs = []
    for i in range(len(slots) - 1):
        arcs.append((slots[i], slots[i + 1]))
        arcs.append((slots[i + 1], slots[i]))

    queue = list(arcs)

    def revise(xi, xj):
        revised = False
        to_remove = []
        for x in domains[xi]:
            # Check if there exists at least one value in xj's domain
            # consistent with x (i.e., different from x for no-back-to-back)
            if not any(y != x for y in domains[xj]):
                to_remove.append(x)
                revised = True
        for x in to_remove:
            domains[xi].remove(x)
        return revised

    while queue:
        xi, xj = queue.pop(0)
        if revise(xi, xj):
            if not domains[xi]:
                return False, domains
            # Add all arcs (xk, xi) where xk != xj
            for i in range(len(slots) - 1):
                if slots[i] == xi and i + 1 < len(slots) and slots[i + 1] != xj:
                    queue.append((slots[i + 1], xi))
                if slots[i + 1] == xi and slots[i] != xj:
                    queue.append((slots[i], xi))

    return True, domains


# MAIN EXECUTION

if __name__ == "__main__":
    try:
        bots, slots, unary = read_csp_input("input.txt")
    except:
        bots = ['A', 'B', 'C']
        slots = ['Slot1', 'Slot2', 'Slot3', 'Slot4']
        unary = {'Slot4': ['C']}

    print("="*60)
    print("QUESTION 2: CONSTRAINT SATISFACTION PROBLEM")
    print("="*60)
    print(f"\nVariables: {slots}")
    print(f"Domain: {bots}")
    print(f"Unary Constraints: {unary}")

    solver = CSPSolver(bots, slots, unary)
    result = solver.solve()

    print(f"\n{'='*50}")
    print(f"Result: {'SUCCESS' if result['success'] else 'FAILURE'}")
    print(f"Heuristic: {result['heuristic']}")
    print(f"Inference: {result['inference']}")
    print(f"Constraints Applied:")
    for c in result['constraints']:
        print(f"  - {c}")
    if result['assignment']:
        print(f"\nFinal Assignment:")
        for slot in slots:
            print(f"  {slot} -> Bot {result['assignment'][slot]}")
    print(f"\nTotal Assignments: {result['total_assignments']}")
    print(f"Total Time: {result['time']:.6f} seconds")

    # AC-3 check
    print(f"\n{'='*50}")
    print("AC-3 Arc Consistency Check")
    print(f"{'='*50}")
    consistent, reduced_domains = ac3_check(bots, slots, unary)
    print(f"Arc Consistent: {consistent}")
    print("Reduced Domains:")
    for slot in slots:
        print(f"  {slot}: {reduced_domains[slot]}")
