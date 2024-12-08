import numpy as np

class DEALS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.F = 0.8  # Mutation factor
        self.CR = 0.9  # Crossover probability
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.scores = np.full(self.population_size, np.inf)
        self.adaptive_local_search = True
        self.dynamic_F = lambda evals: 0.5 + 0.3 * np.cos(np.pi * evals / self.budget)
        self.dynamic_CR = lambda evals: 0.9 - 0.4 * np.sin(np.pi * evals / self.budget)  # Adaptive Crossover

    def __call__(self, func):
        eval_count = 0
        # Initial evaluation
        for i in range(self.population_size):
            self.scores[i] = func(self.population[i])
            eval_count += 1
            if eval_count >= self.budget:
                return self.best_solution()

        while eval_count < self.budget:
            for i in range(self.population_size):
                a, b, c = self.select_others(i)
                current_F = self.dynamic_F(eval_count)
                mutant = self.mutate(a, b, c, current_F)
                trial = self.crossover(self.population[i], mutant, eval_count)
                trial_score = func(trial)
                eval_count += 1
                if trial_score < self.scores[i]:
                    self.population[i] = trial
                    self.scores[i] = trial_score

                if eval_count >= self.budget:
                    return self.best_solution()

            if self.adaptive_local_search:
                self.local_search(func, eval_count)
                
        return self.best_solution()

    def select_others(self, index):
        indices = list(range(self.population_size))
        indices.remove(index)
        selected = np.random.choice(indices, 3, replace=False)
        return self.population[selected[0]], self.population[selected[1]], self.population[selected[2]]

    def mutate(self, a, b, c, F):
        mutation = a + F * (b - c)
        return np.clip(mutation, self.lower_bound, self.upper_bound)

    def crossover(self, target, mutant, eval_count):
        current_CR = self.dynamic_CR(eval_count)
        crossover_mask = np.random.rand(self.dim) < current_CR
        if not np.any(crossover_mask):
            crossover_mask[np.random.randint(0, self.dim)] = True  # Ensure at least one crossover
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def local_search(self, func, eval_count):
        best_idx = np.argmin(self.scores)
        best = self.population[best_idx]
        step_size = (self.upper_bound - self.lower_bound) * 0.03  # Slightly smaller step size
        for _ in range(7):  # More local search steps
            if eval_count >= self.budget:
                break
            local_steps = np.random.uniform(-step_size, step_size, self.dim)
            candidate = np.clip(best + local_steps, self.lower_bound, self.upper_bound)
            candidate_score = func(candidate)
            eval_count += 1
            if candidate_score < self.scores[best_idx]:
                self.population[best_idx] = candidate
                self.scores[best_idx] = candidate_score
                best = candidate

    def best_solution(self):
        best_idx = np.argmin(self.scores)
        return self.population[best_idx], self.scores[best_idx]