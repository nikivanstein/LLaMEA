import numpy as np

class EnhancedSynergizedMemeticDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = max(20, 6 * dim)  # Increased population size
        self.lb = -5.0
        self.ub = 5.0
        self.F = 0.5
        self.CR = 0.8  # Reduced crossover rate for diversity
        self.population = np.random.uniform(self.lb, self.ub, (self.pop_size, dim))
        self.best_solution = None
        self.best_fitness = np.inf
        self.evaluation_count = 0
        self.strategy_success = np.zeros(4)  # Added new strategy
        self.strategy_prob = np.ones(4) / 4  # Adjusted probability for four strategies
        self.adaptive_factor = 0.08  # Slightly reduced adaptation factor
        self.mutation_probability = 0.03  # Reduced mutation probability
        self.memory = {}
        self.learning_rate = 0.08  # Adjusted learning rate
        self.elite_memory = []  # New memory for elite solutions

    def select_strategy(self):
        if np.random.rand() < self.learning_rate:
            return np.random.choice(4)  # Include new strategy
        return np.random.choice(4, p=self.strategy_prob)

    def mutate(self, idx):
        if np.random.rand() < self.mutation_probability:
            return self.random_mutation(idx), 0
        strategies = [self.rand_1, self.rand_2, self.current_to_best_with_elite, self.elite_guided]  # Added new strategy
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
        elite_size = int(0.25 * self.pop_size)  # Increased elite size
        if idx in self.memory:
            best_elite = self.memory[idx]
        else:
            elite_indices = np.argsort([self.func(ind) for ind in self.population])[:elite_size]
            best_elite = self.population[np.random.choice(elite_indices)]
            self.memory[idx] = best_elite
        candidates = [i for i in range(self.pop_size) if i != idx]
        r1, r2 = np.random.choice(candidates, 2, replace=False)
        return self.population[idx] + self.F * (best_elite - self.population[idx]) + self.F * (self.population[r1] - self.population[r2])

    def elite_guided(self, idx):
        if not self.elite_memory:
            return self.rand_1(idx)
        elite_idx = np.random.choice(len(self.elite_memory))
        elite_solution = self.elite_memory[elite_idx]
        candidates = [i for i in range(self.pop_size) if i != idx]
        r1 = np.random.choice(candidates)
        return elite_solution + self.F * (self.population[r1] - elite_solution)

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
            self.elite_memory.append(solution)  # Add to elite memory
            if len(self.elite_memory) > 5:  # Limit size of elite memory
                self.elite_memory.pop(0)
        return fitness

    def adapt_params(self):
        self.F = np.clip(self.F + self.adaptive_factor * (np.random.rand() - 0.5), 0.4, 0.9)
        self.CR = np.clip(self.CR + self.adaptive_factor * (np.random.rand() - 0.5), 0.7, 0.9)
        self.learning_rate = np.clip(self.learning_rate + self.adaptive_factor * (np.random.rand() - 0.5), 0.01, 0.1)
        total_success = np.sum(self.strategy_success)
        self.strategy_prob = self.strategy_success / total_success if total_success > 0 else np.ones(4) / 4  # Adjust probability

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