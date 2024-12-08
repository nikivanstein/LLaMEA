import numpy as np

class DEALS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 10 * dim
        self.dynamic_population_size = lambda evals: max(4, int(self.initial_population_size * (1 - evals / self.budget)))
        self.F = 0.8  # Mutation factor
        self.CR = 0.9  # Crossover probability
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.initial_population_size, dim))
        self.scores = np.full(self.initial_population_size, np.inf)
        self.adaptive_local_search = True
        self.dynamic_F = lambda evals: 0.5 + 0.3 * np.cos(np.pi * evals / self.budget)

    def __call__(self, func):
        eval_count = 0
        # Initial evaluation
        for i in range(self.initial_population_size):
            self.scores[i] = func(self.population[i])
            eval_count += 1
            if eval_count >= self.budget:
                return self.best_solution()

        while eval_count < self.budget:
            current_population_size = self.dynamic_population_size(eval_count)
            for i in range(current_population_size):
                a, b, c = self.select_others(i, current_population_size)
                current_F = self.dynamic_F(eval_count)
                mutant = self.mutate(a, b, c, current_F)
                trial = self.crossover(self.population[i], mutant)
                trial_score = func(trial)
                eval_count += 1
                if trial_score < self.scores[i]:
                    self.population[i] = trial
                    self.scores[i] = trial_score

                if eval_count >= self.budget:
                    return self.best_solution()

            if self.adaptive_local_search:
                self.local_search(func, eval_count, current_population_size)

        return self.best_solution()

    def select_others(self, index, pop_size):
        indices = list(range(pop_size))
        indices.remove(index)
        selected = np.random.choice(indices, 3, replace=False)
        return self.population[selected[0]], self.population[selected[1]], self.population[selected[2]]

    def mutate(self, a, b, c, F):
        mutation = a + F * (b - c)
        return np.clip(mutation, self.lower_bound, self.upper_bound)

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.CR
        if not np.any(crossover_mask):
            crossover_mask[np.random.randint(0, self.dim)] = True  # Ensure at least one crossover
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def local_search(self, func, eval_count, pop_size):
        best_idx = np.argmin(self.scores[:pop_size])
        best = self.population[best_idx]
        step_size = (self.upper_bound - self.lower_bound) * 0.05
        for _ in range(5):  # Limited local search steps
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