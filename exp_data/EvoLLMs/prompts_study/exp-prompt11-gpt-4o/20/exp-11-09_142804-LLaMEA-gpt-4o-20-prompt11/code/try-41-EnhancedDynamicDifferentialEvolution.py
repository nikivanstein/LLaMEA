import numpy as np

class EnhancedDynamicDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.crossover_rate = 0.85  # Slightly reduced for balanced exploration-exploitation
        self.mutation_factor = 0.9  # Increased mutation factor for broader search
        self.population_scaling = 0.5 + (0.3 * np.random.rand())
        self.population_size = int(self.budget / (5 * dim))  # Adjusted to increase population diversity
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.max_local_search_iterations = 5  # Increased local search iterations for thorough exploration
        self.local_search_probability = 0.45  # Enhanced probability for exploiting local search

    def _evaluate_population(self, func):
        return np.array([func(ind) for ind in self.population])

    def _mutate(self, target_idx):
        indices = [idx for idx in range(self.population_size) if idx != target_idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        adaptive_mutation_factor = self.mutation_factor * (1 + np.random.uniform(-0.15, 0.15))  # More variability in mutation
        mutant_vector = self.population[a] + self.population_scaling * adaptive_mutation_factor * (
            self.population[b] - self.population[c])
        mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)
        return mutant_vector

    def _crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.crossover_rate
        if not np.any(crossover_mask):
            crossover_mask[np.random.randint(0, self.dim)] = True
        trial_vector = np.where(crossover_mask, mutant, target)
        return trial_vector

    def _adaptive_selection(self, target_score, trial_score, idx, trial):
        if trial_score < target_score:
            self.population[idx] = trial
            return trial_score
        if np.random.rand() < 0.2:  # Add stochastic acceptance factor
            self.population[idx] = trial
            return trial_score
        return target_score

    def _stochastic_local_search(self, solution, func, max_iters):
        best_solution = solution.copy()
        best_score = func(best_solution)
        for _ in range(max_iters):
            perturbation = np.random.normal(0.0, 0.1, self.dim)
            candidate_solution = np.clip(solution + perturbation, self.lower_bound, self.upper_bound)
            candidate_score = func(candidate_solution)
            if candidate_score < best_score:
                best_solution = candidate_solution
                best_score = candidate_score
        return best_solution

    def __call__(self, func):
        population_scores = self._evaluate_population(func)
        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                target = self.population[i]
                mutant = self._mutate(i)
                trial = self._crossover(target, mutant)
                trial_score = func(trial)
                evaluations += 1

                population_scores[i] = self._adaptive_selection(population_scores[i], trial_score, i, trial)

                if np.random.rand() < self.local_search_probability:
                    improved_solution = self._stochastic_local_search(trial, func, self.max_local_search_iterations)
                    improved_score = func(improved_solution)
                    evaluations += 1

                    if improved_score < population_scores[i]:
                        self.population[i] = improved_solution
                        population_scores[i] = improved_score

        best_index = np.argmin(population_scores)
        return self.population[best_index]