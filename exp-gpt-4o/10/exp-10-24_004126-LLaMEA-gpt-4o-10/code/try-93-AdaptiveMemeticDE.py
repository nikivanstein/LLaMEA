import numpy as np

class AdaptiveMemeticDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = max(10, 6 * dim)
        self.lb = -5.0
        self.ub = 5.0
        self.F = 0.5
        self.CR = 0.9
        self.population = np.random.uniform(self.lb, self.ub, (self.pop_size, dim))
        self.best_solution = None
        self.best_fitness = np.inf
        self.evaluation_count = 0
        self.strategy_success = np.zeros(3)
        self.strategy_prob = np.ones(3) / 3
        self.adaptive_factor = 0.05
        self.mutation_probability = 0.1
        self.memory = {}
        self.learning_rate = 0.02

    def select_strategy(self):
        if np.random.rand() < self.learning_rate:
            return np.random.choice(3)
        return np.random.choice(3, p=self.strategy_prob)

    def mutate(self, idx):
        if np.random.rand() < self.mutation_probability:
            return self.random_mutation(idx), 0
        strategies = [self.rand_1, self.rand_2, self.current_to_best_with_memory_elite]
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

    def current_to_best_with_memory_elite(self, idx):
        elite_size = int(0.1 * self.pop_size)
        if idx in self.memory:
            best_elite = self.memory[idx]
        else:
            elite_indices = np.argsort([self.func(ind) for ind in self.population])[:elite_size]
            best_elite = self.population[np.random.choice(elite_indices)]
            self.memory[idx] = best_elite
        candidates = [i for i in range(self.pop_size) if i != idx]
        r1, r2 = np.random.choice(candidates, 2, replace=False)
        return self.population[idx] + self.F * (best_elite - self.population[idx]) + self.F * (self.population[r1] - self.population[r2])

    def random_mutation(self, idx):
        return np.random.uniform(self.lb, self.ub, self.dim)

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
            self.best_solution = np.copy(solution)
        return fitness

    def adapt_params(self):
        self.F = np.clip(self.F + self.adaptive_factor * (np.random.rand() - 0.5), 0.4, 0.9)
        self.CR = np.clip(self.CR + self.adaptive_factor * (np.random.rand() - 0.5), 0.8, 1.0)
        self.learning_rate = np.clip(self.learning_rate + self.adaptive_factor * (np.random.rand() - 0.5), 0.01, 0.05)
        total_success = np.sum(self.strategy_success)
        self.strategy_prob = self.strategy_success / total_success if total_success > 0 else np.ones(3) / 3

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
                    self.strategy_success[strat_index] += 1
                else:
                    self.strategy_success[strat_index] *= 0.95
            self.adapt_params()
        return self.best_solution, self.best_fitness