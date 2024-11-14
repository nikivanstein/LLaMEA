import numpy as np

class ModifiedAdaptiveDEPS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 20 + int(3.0 * np.sqrt(self.dim))
        self.global_best = None
        self.best_cost = float('inf')
        self.init_population_size = self.population_size
        self.memory = []

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evals = self.population_size
        F = 0.5 + 0.3 * np.random.rand()
        CR = 0.8 + 0.1 * np.random.rand()

        while evals < self.budget:
            self.population_size = max(5, self.init_population_size - int(evals / self.budget * (self.init_population_size - 5)))
            new_memory = []
            
            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), self.lower_bound, self.upper_bound)
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                trial_cost = func(trial)
                evals += 1
                if trial_cost < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_cost
                    new_memory.append((F, CR))
                    if trial_cost < self.best_cost:
                        self.global_best = trial
                        self.best_cost = trial_cost
                if evals >= self.budget:
                    break

            if new_memory:
                F = np.mean([mem[0] for mem in new_memory])
                CR = np.mean([mem[1] for mem in new_memory])
            self.memory.extend(new_memory)
            if len(self.memory) > 10:
                self.memory = self.memory[-10:]

        return self.global_best