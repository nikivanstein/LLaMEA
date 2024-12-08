import numpy as np

class HybridHarmonyDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.hms = 10  # Harmony Memory Size
        self.hmcr = 0.9  # Harmony Memory Considering Rate
        self.par = 0.3  # Pitch Adjustment Rate
        self.f = 0.5  # Differential Evolution scaling factor
        self.cr = 0.9  # Crossover rate
        self.harmony_memory = np.random.uniform(self.lb, self.ub, (self.hms, dim))
        self.hm_fitness = np.full(self.hms, np.inf)
        self.eval_count = 0

    def evaluate(self, solution, func):
        if self.eval_count < self.budget:
            self.eval_count += 1
            return func(solution)
        else:
            return np.inf

    def update_harmony_memory(self, candidate, candidate_fitness):
        worst_index = np.argmax(self.hm_fitness)
        if candidate_fitness < self.hm_fitness[worst_index]:
            self.harmony_memory[worst_index] = candidate
            self.hm_fitness[worst_index] = candidate_fitness

    def new_harmony(self):
        new_solution = np.zeros(self.dim)
        for i in range(self.dim):
            if np.random.rand() < self.hmcr:
                new_solution[i] = self.harmony_memory[np.random.randint(self.hms), i]
                if np.random.rand() < self.par:
                    new_solution[i] += np.random.uniform(-0.5, 0.5)  # Slight adjustment
            else:
                new_solution[i] = np.random.uniform(self.lb, self.ub)
        return np.clip(new_solution, self.lb, self.ub)

    def differential_evolution(self):
        candidates = np.random.choice(self.hms, 3, replace=False)
        a, b, c = self.harmony_memory[candidates]
        mutant = np.clip(a + self.f * (b - c), self.lb, self.ub)
        trial = np.copy(mutant)
        j_rand = np.random.randint(self.dim)
        for j in range(self.dim):
            if np.random.rand() > self.cr and j != j_rand:
                trial[j] = a[j]
        # Dynamic adjustment of crossover rate based on harmony memory fitness diversity
        fitness_range = np.max(self.hm_fitness) - np.min(self.hm_fitness)
        self.cr = 0.9 - 0.5 * (fitness_range / (np.abs(np.min(self.hm_fitness)) + 1e-9))
        return trial

    def __call__(self, func):
        for i in range(self.hms):
            self.hm_fitness[i] = self.evaluate(self.harmony_memory[i], func)

        while self.eval_count < self.budget:
            if np.random.rand() < 0.5:
                new_solution = self.new_harmony()
            else:
                new_solution = self.differential_evolution()
            
            new_fitness = self.evaluate(new_solution, func)
            self.update_harmony_memory(new_solution, new_fitness)

        best_index = np.argmin(self.hm_fitness)
        return self.harmony_memory[best_index]