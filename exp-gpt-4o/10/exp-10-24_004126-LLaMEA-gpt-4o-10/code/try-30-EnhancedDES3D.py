import numpy as np

class EnhancedDES3D:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = max(10, 5 * dim)
        self.lb = -5.0
        self.ub = 5.0
        self.F = 0.5
        self.CR = 0.9
        self.population = np.random.uniform(self.lb, self.ub, (self.pop_size, dim))
        self.best_solution = None
        self.best_fitness = np.inf
        self.evaluation_count = 0
        self.history = np.zeros(3)
        self.func = None
        self.success_rates = np.zeros(3)
        self.elite_fraction = 0.2  # Preserve top 20% elite solutions
        self.dynamic_F = (0.4, 0.9)  # Range for dynamic F

    def select_strategy(self):
        total_success = np.sum(self.success_rates)
        probabilities = self.success_rates / total_success if total_success > 0 else np.ones(3) / 3
        return np.random.choice(3, p=probabilities)

    def mutate(self, idx):
        strategies = [self.rand_1, self.rand_2, self.current_to_best_with_elite]
        strat_index = self.select_strategy()
        mutant = strategies[strat_index](idx)
        return mutant, strat_index

    def rand_1(self, idx):
        candidates = [i for i in range(self.pop_size) if i != idx]
        r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
        return self.population[r1] + self.F * (self.population[r2] - self.population[r3])

    def rand_2(self, idx):
        candidates = [i for i in range(self.pop_size) if i != idx]
        r1, r2, r3, r4, r5 = np.random.choice(candidates, 5, replace=False)
        return self.population[r1] + self.F * (self.population[r2] - self.population[r3]) + self.F * (self.population[r4] - self.population[r5])

    def current_to_best_with_elite(self, idx):
        elite_size = int(self.elite_fraction * self.pop_size)
        elite_indices = np.argsort([self.func(ind) for ind in self.population])[:elite_size]
        best_elite = self.population[np.random.choice(elite_indices)]
        candidates = [i for i in range(self.pop_size) if i != idx]
        r1, r2 = np.random.choice(candidates, 2, replace=False)
        return self.population[idx] + self.F * (best_elite - self.population[idx]) + self.F * (self.population[r1] - self.population[r2])

    def crossover(self, target, mutant):
        trial = np.copy(target)
        j_rand = np.random.randint(0, self.dim)
        for j in range(self.dim):
            if np.random.rand() < self.CR or j == j_rand:
                trial[j] = mutant[j]
        return trial

    def evaluate(self, solution):
        if self.evaluation_count >= self.budget:
            raise RuntimeError("Budget exceeded")
        fitness = self.func(solution)
        self.evaluation_count += 1
        if fitness < self.best_fitness:
            self.best_fitness = fitness
            self.best_solution = solution
        return fitness

    def dynamic_parameter_adjustment(self):
        # Adjust F dynamically based on progress
        progress = self.evaluation_count / self.budget
        self.F = self.dynamic_F[0] + (self.dynamic_F[1] - self.dynamic_F[0]) * (1 - progress)

    def __call__(self, func):
        self.func = func
        while self.evaluation_count < self.budget:
            self.dynamic_parameter_adjustment()
            for i in range(self.pop_size):
                target = self.population[i]
                mutant, strat_index = self.mutate(i)
                mutant = np.clip(mutant, self.lb, self.ub)
                trial = self.crossover(target, mutant)
                trial_fitness = self.evaluate(trial)
                target_fitness = self.evaluate(target)
                if trial_fitness < target_fitness:
                    self.population[i] = trial
                    self.history[strat_index] += 1
                    self.success_rates[strat_index] += 1
                else:
                    self.success_rates[strat_index] *= 0.95
        return self.best_solution, self.best_fitness