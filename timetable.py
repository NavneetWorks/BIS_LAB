"""
Gene Expression Algorithm (GEA) - College Timetable Scheduling
CSE-A aur CSE-B, 6 subjects, 6 slots.

Chromosome (genotype): [A1..A6 | B1..B6]  -> har number ek subject hai
Gene expression: genotype -> phenotype (slot -> subject -> teacher, room)
Fitness = 1000 - 20*T - 20*R
"""

import random

# ----------------------------- 1. PROBLEM DATA -----------------------------
SLOTS = [
    ("S1", "Monday", "9-10"),
    ("S2", "Monday", "10-11"),
    ("S3", "Monday", "11-12"),
    ("S4", "Tuesday", "9-10"),
    ("S5", "Tuesday", "10-11"),
    ("S6", "Tuesday", "11-12"),
]

# gene -> (subject, teacher, room)
SUBJECTS = {
    1: ("DSA", "T1", "R1"),
    2: ("DBMS", "T2", "R2"),
    3: ("OS", "T3", "R1"),
    4: ("Maths", "T4", "R2"),
    5: ("CN", "T5", "R1"),
    6: ("SE", "T6", "R2"),
}

N = len(SLOTS)  # 6 slots / 6 subjects per class

# ----------------------------- 2. PARAMETERS -------------------------------
POP_SIZE = 4
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.1
MAX_GENERATIONS = 50   # example mein 2 tha; random run mein zyada generations safe hain
MAX_FITNESS = 1000
PENALTY = 20


# ----------------------------- 3. POPULATION -------------------------------
def create_chromosome():
    """CSE-A fixed (1..6), CSE-B random permutation (har subject exactly ek baar)."""
    a = list(range(1, N + 1))
    b = random.sample(range(1, N + 1), N)
    return a + b


def init_population(size):
    return [create_chromosome() for _ in range(size)]


# ----------------------------- GENE EXPRESSION -----------------------------
def express(chromosome):
    """Genotype (numbers) -> phenotype (actual timetable rows)."""
    a, b = chromosome[:N], chromosome[N:]
    table = []
    for i, (slot, day, time) in enumerate(SLOTS):
        table.append({
            "slot": slot, "day": day, "time": time,
            "A": SUBJECTS[a[i]],   # (subject, teacher, room)
            "B": SUBJECTS[b[i]],
        })
    return table


# ----------------------------- 4. FITNESS ----------------------------------
def count_conflicts(chromosome):
    t = r = 0
    for row in express(chromosome):
        _, teacher_a, room_a = row["A"]
        _, teacher_b, room_b = row["B"]
        if teacher_a == teacher_b:
            t += 1
        if room_a == room_b:
            r += 1
    return t, r


def fitness(chromosome):
    t, r = count_conflicts(chromosome)
    return MAX_FITNESS - PENALTY * t - PENALTY * r


# ----------------------------- 5. SELECTION --------------------------------
def roulette_select(population, fits):
    """Probability = fitness / total fitness."""
    total = sum(fits)
    pick = random.uniform(0, total)
    cumulative = 0
    for chrom, f in zip(population, fits):
        cumulative += f
        if pick <= cumulative:
            return chrom
    return population[-1]


# ----------------------------- 6. CROSSOVER (OX) ---------------------------
def order_crossover_part(p_seg, p_rest, lo, hi):
    """
    p_seg se positions lo..hi copy karo, baaki positions p_rest ke genes se
    (unke original order mein) left-to-right bharo.
    """
    child = [None] * N
    child[lo:hi + 1] = p_seg[lo:hi + 1]
    used = set(child[lo:hi + 1])
    remaining = [g for g in p_rest if g not in used]
    it = iter(remaining)
    for i in range(N):
        if child[i] is None:
            child[i] = next(it)
    return child


def crossover(parent1, parent2):
    """Sirf CSE-B part par OX (CSE-A dono parents mein same hai)."""
    if random.random() >= CROSSOVER_RATE:
        return parent1[:], parent2[:]          # crossover nahi hua

    b1, b2 = parent1[N:], parent2[N:]
    lo, hi = sorted(random.sample(range(N), 2))
    child1_b = order_crossover_part(b2, b1, lo, hi)   # segment parent2 se
    child2_b = order_crossover_part(b1, b2, lo, hi)   # segment parent1 se
    return parent1[:N] + child1_b, parent2[:N] + child2_b


# ----------------------------- 7. MUTATION (SWAP) --------------------------
def mutate(chromosome):
    child = chromosome[:]
    if random.random() < MUTATION_RATE:
        i, j = random.sample(range(N, 2 * N), 2)   # CSE-B ke do positions
        child[i], child[j] = child[j], child[i]
    return child


# ----------------------------- 8-9. GEA LOOP -------------------------------
def run_gea(verbose=True):
    population = init_population(POP_SIZE)
    best = max(population, key=fitness)

    for gen in range(1, MAX_GENERATIONS + 1):
        fits = [fitness(c) for c in population]
        best = max(population, key=fitness)

        if verbose:
            print(f"Generation {gen}: best fitness = {fitness(best)}  {best[:N]} | {best[N:]}")

        # convergence: T = 0 aur R = 0
        if fitness(best) == MAX_FITNESS:
            if verbose:
                print(f"\nOptimal solution Generation {gen} mein mil gaya.")
            break

        # naya population (elitism: best chromosome seedha next generation mein)
        new_pop = [best[:]]
        while len(new_pop) < POP_SIZE:
            p1 = roulette_select(population, fits)
            p2 = roulette_select(population, fits)
            c1, c2 = crossover(p1, p2)
            new_pop.append(mutate(c1))
            if len(new_pop) < POP_SIZE:
                new_pop.append(mutate(c2))
        population = new_pop

    best = max(population, key=fitness)
    return best


# ----------------------------- 10. OUTPUT ----------------------------------
def print_timetable(chromosome):
    t, r = count_conflicts(chromosome)
    print(f"\nBest chromosome: {chromosome[:N]} | {chromosome[N:]}")
    print(f"Teacher conflicts = {t}, Room conflicts = {r}, Fitness = {fitness(chromosome)}\n")

    header = f"{'Day':<9}{'Time':<7}| {'CSE-A':<7}{'Teacher':<8}{'Room':<5}| {'CSE-B':<7}{'Teacher':<8}{'Room':<5}"
    print(header)
    print("-" * len(header))
    for row in express(chromosome):
        sa, ta, ra = row["A"]
        sb, tb, rb = row["B"]
        print(f"{row['day']:<9}{row['time']:<7}| {sa:<7}{ta:<8}{ra:<5}| {sb:<7}{tb:<8}{rb:<5}")


if __name__ == "__main__":
    # reproducible output chahiye to seed uncomment karo:
    # random.seed(42)
    best = run_gea()
    print_timetable(best)
