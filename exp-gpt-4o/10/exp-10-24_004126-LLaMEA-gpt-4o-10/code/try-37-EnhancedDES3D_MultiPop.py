import numpy as np

class EnhancedDES3D_MultiPop:
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
        self.elite_fraction = 0.2
        self.adaptive_factor = 0.1
        self.num_subpops = 3
        self.subpopulations = [self.population[i::self.num_subpops] for i in range(self.num_subpops)]

    def select_strategy(self):
        total_success = np.sum(self.success_rates)
        probabilities = self.success_rates / total_success if total_success > 0 else np.ones(3) / 3
        return np.random.choice(3, p=probabilities)

    def mutate(self, idx, subpop):
        strategies = [self.rand_1, self.rand_2, self.current_to_best_with_elite]
        strat_index = self.select_strategy()
        mutant = strategies[strat_index](idx, subpop)
        return mutant, strat_index

    def rand_1(self, idx, subpop):
        candidates = [i for i in range(len(subpop)) if i != idx]
        r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
        return subpop[r1] + self.F * (subpop[r2] - subpop[r3])

    def rand_2(self, idx, subpop):
        candidates = [i for i in range(len(subpop)) if i != idx]
        r1, r2, r3, r4, r5 = np.random.choice(candidates, 5, replace=False)
        return subpop[r1] + self.F * (subpop[r2] - subpop[r3]) + self.F * (subpop[r4] - subpop[r5])

    def current_to_best_with_elite(self, idx, subpop):
        elite_size = int(self.elite_fraction * len(subpop))
        elite_indices = np.argsort([self.func(ind) for ind in subpop])[:elite_size]
        best_elite = subpop[np.random.choice(elite_indices)]
        candidates = [i for i in range(len(subpop)) if i != idx]
        r1, r2 = np.random.choice(candidates, 2, replace=False)
        return subpop[idx] + self.F * (best_elite - subpop[idx]) + self.F * (subpop[r1] - subpop[r2])

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

    def __call__(self, func):
        self.func = func
        while self.evaluation_count < self.budget:
            for subpop in self.subpopulations:
                for i in range(len(subpop)):
                    target = subpop[i]
                    mutant, strat_index = self.mutate(i, subpop)
                    mutant = np.clip(mutant, self.lb, self.ub)
                    trial = self.crossover(target, mutant)
                    trial_fitness = self.evaluate(trial)
                    target_fitness = self.evaluate(target)
                    if trial_fitness < target_fitness:
                        subpop[i] = trial
                        self.history[strat_index] += 1
                        self.success_rates[strat_index] += 1
                    else:
                        self.success_rates[strat_index] *= 0.95
            self.adapt_params()
        return self.best_solution, self.best_fitness