import numpy as np

class EnhancedDynamicDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.mutation_factor = 0.85
        self.population_scaling = 0.5 + (0.3 * np.random.rand())
        self.population_size = int(self.budget / (6 * dim))
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.adaptive_crossover_rate = np.linspace(0.6, 0.9, self.population_size)
        self.dynamic_mutation_factor = np.logspace(-2, 0, num=self.population_size)
        self.dynamic_local_search_iters = np.random.randint(3, 8, self.population_size)  # Adjusted range

    def _evaluate_population(self, func):
        return np.array([func(ind) for ind in self.population])

    def _mutate(self, target_idx):
        indices = [idx for idx in range(self.population_size) if idx != target_idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        learning_rate = self.dynamic_mutation_factor[target_idx]
        adaptive_mutation_factor = self.mutation_factor * (1 + np.random.uniform(-0.1, 0.15)) * learning_rate  # Slightly increased range
        mutant_vector = self.population[a] + self.population_scaling * adaptive_mutation_factor * (
            self.population[b] - self.population[c])
        mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)
        return mutant_vector
    
    def _crossover(self, target, mutant, idx):
        crossover_rate = self.adaptive_crossover_rate[idx]
        crossover_mask = np.random.rand(self.dim) < crossover_rate
        if not np.any(crossover_mask):
            crossover_mask[np.random.randint(0, self.dim)] = True
        trial_vector = np.where(crossover_mask, mutant, target)
        return trial_vector

    def _trend_aware_local_search(self, solution, func, max_iters):
        best_solution = solution.copy()
        best_score = func(best_solution)
        trend_factor = 0.8  # Integrating trend factor
        for _ in range(max_iters):
            perturbation = np.random.normal(0.0, 0.1, self.dim) * trend_factor
            candidate_solution = np.clip(solution + perturbation, self.lower_bound, self.upper_bound)
            candidate_score = func(candidate_solution)
            if candidate_score < best_score:
                best_solution = candidate_solution
                best_score = candidate_score
        return best_solution

    def _multi_trajectory_reinforcement(self):
        path_choices = np.random.uniform(-0.5, 0.5, (self.population_size, self.dim))
        return path_choices

    def __call__(self, func):
        population_scores = self._evaluate_population(func)
        evaluations = self.population_size

        multi_trajectory_population = self._multi_trajectory_reinforcement()
        evaluations += self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                target = self.population[i]
                mutant = self._mutate(i)
                trial = self._crossover(target, mutant, i)
                trial_score = func(trial)
                evaluations += 1

                if trial_score < population_scores[i]:
                    self.population[i] = trial
                    population_scores[i] = trial_score

                    if np.random.rand() < self.dynamic_mutation_factor[i]:
                        improved_solution = self._trend_aware_local_search(trial, func, self.dynamic_local_search_iters[i])
                        improved_score = func(improved_solution)
                        evaluations += 1

                        if improved_score < trial_score:
                            self.population[i] = improved_solution
                            population_scores[i] = improved_score

        best_index = np.argmin(population_scores)
        return self.population[best_index]