import numpy as np

class EnhancedDES3D:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = max(20, 7 * dim)  # Increased population size for more diversity
        self.lb = -5.0
        self.ub = 5.0
        self.F = 0.6  # Slightly higher mutation factor for exploration
        self.CR = 0.8  # Slightly lower crossover rate for more exploitation
        self.population = np.random.uniform(self.lb, self.ub, (self.pop_size, dim))
        self.best_solution = None
        self.best_fitness = np.inf
        self.evaluation_count = 0
        self.history = np.zeros(3)
        self.func = None
        self.success_rates = np.ones(3)  # Start with a non-zero success rate

    def select_strategy(self):
        total_success = np.sum(self.success_rates)
        if total_success > 0:
            probabilities = self.success_rates / total_success
        else:
            probabilities = np.ones(3) / 3
        return np.random.choice(3, p=probabilities)

    def mutate(self, idx):
        strategies = [self.rand_1, self.rand_2, self.current_to_best]
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

    def current_to_best(self, idx):
        best_index = np.argmin([self.func(ind) for ind in self.population])
        candidates = [i for i in range(self.pop_size) if i != idx]
        r1, r2 = np.random.choice(candidates, 2, replace=False)
        return self.population[idx] + self.F * (self.population[best_index] - self.population[idx]) + self.F * (self.population[r1] - self.population[r2])

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

    def __call__(self, func):
        self.func = func
        while self.evaluation_count < self.budget:
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
                    self.success_rates[strat_index] += 0.1  # Slightly increase the success rate gain
                else:
                    self.success_rates[strat_index] *= 0.9  # Increase decay factor for faster adaptation
        return self.best_solution, self.best_fitness