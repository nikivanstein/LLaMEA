import numpy as np

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.cross_prob = 0.9 + 0.1 / dim
        self.diff_weight = 0.5 + np.random.rand() * 0.5
        self.current_budget = 0
        self.success_rate_history = []  # New line for tracking success rates

    def _initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))

    def _mutate(self, population):
        idxs = np.random.choice(self.population_size, 3, replace=False)
        a, b, c = population[idxs]
        success_rate = np.random.rand()
        adaptive_diff_weight = self.diff_weight * (1 - success_rate)
        if success_rate > 0.5:
            adaptive_diff_weight *= 1.2  # Slightly decrease the inflation factor from 1.25 to 1.2
        if self.success_rate_history:  # Modified to adjust mutation weight based on success history
            historical_success_rate = np.mean(self.success_rate_history[-min(len(self.success_rate_history), 10):])  # New line
            success_rate_variance = np.var(self.success_rate_history[-min(len(self.success_rate_history), 10):])  # New line
            adaptive_diff_weight *= (1 + 0.2 * historical_success_rate + 0.05 * success_rate_variance)  # Slightly reduce variance influence from 0.1 to 0.05
        mutant = a + adaptive_diff_weight * (b - c)
        return np.clip(mutant, self.lower_bound, self.upper_bound)

    def _crossover(self, target, mutant):
        use_binomial = np.random.rand() < 0.5
        adaptive_cross_prob = self.cross_prob * (0.8 + 0.2 * np.random.rand())
        if use_binomial:
            crossover_mask = np.random.rand(self.dim) < adaptive_cross_prob
            return np.where(crossover_mask, mutant, target)
        else:
            return adaptive_cross_prob * mutant + (1 - adaptive_cross_prob) * target

    def _directional_search(self, current, func):
        direction = np.random.normal(size=self.dim)
        direction /= np.linalg.norm(direction)
        decay_factor = 0.9 ** (self.current_budget / self.budget)  # Slightly increase decay factor from 0.85 to 0.9
        secondary_decay = (1 - self.current_budget/self.budget) ** 2
        budget_ratio = self.current_budget / self.budget
        step_size = (self.upper_bound - self.lower_bound) * (0.1 + 0.1 * (1 - budget_ratio) * decay_factor * secondary_decay * (0.8 + 0.4 * budget_ratio))
        candidate = np.clip(current + step_size * direction, self.lower_bound, self.upper_bound)
        gradient_info = np.random.normal(0, 0.05, size=self.dim)  # Added line for using gradient-like information
        candidate += gradient_info  # Line modified to apply gradient influence
        return candidate if func(candidate) < func(current) else current

    def __call__(self, func):
        population = self._initialize_population()
        fitness = np.array([func(ind) for ind in population])
        self.current_budget += self.population_size

        while self.current_budget < self.budget:
            successes = 0
            for i in range(self.population_size):
                mutant = self._mutate(population)
                trial = self._crossover(population[i], mutant)
                trial = self._directional_search(trial, func)
                trial_fitness = func(trial)
                self.current_budget += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    successes += 1

                if self.current_budget >= self.budget:
                    break

            if successes > 0:
                self.cross_prob = min(1.0, self.cross_prob + 0.01 * (successes / self.population_size))
                self.success_rate_history.append(successes / self.population_size)  # New line to track success rates

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]