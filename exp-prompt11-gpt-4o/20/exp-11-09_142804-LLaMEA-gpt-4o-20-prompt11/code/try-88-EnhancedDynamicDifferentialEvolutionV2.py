import numpy as np

class EnhancedDynamicDifferentialEvolutionV2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.base_mutation_factor = 0.85
        self.mutation_scaling = np.linspace(0.5, 1.0, num=int(budget / (6 * dim)))  # Dynamic scaling factor
        self.population_size = int(self.budget / (6 * dim))
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.adaptive_crossover_rate = np.linspace(0.6, 0.95, self.population_size)  # Increased upper bound
        self.dynamic_local_search_iters = np.random.randint(4, 8, self.population_size)  # Modified iterations

    def _evaluate_population(self, func):
        return np.array([func(ind) for ind in self.population])

    def _mutate(self, target_idx):
        indices = [idx for idx in range(self.population_size) if idx != target_idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        learning_rate = self.mutation_scaling[target_idx]  # Dynamic scaling applied
        adaptive_mutation_factor = self.base_mutation_factor * (1 + np.random.uniform(-0.1, 0.1)) * learning_rate
        mutant_vector = self.population[a] + adaptive_mutation_factor * (self.population[b] - self.population[c])
        mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)
        return mutant_vector
    
    def _crossover(self, target, mutant, idx):
        crossover_rate = self.adaptive_crossover_rate[idx]
        crossover_mask = np.random.rand(self.dim) < crossover_rate
        if not np.any(crossover_mask):
            crossover_mask[np.random.randint(0, self.dim)] = True
        trial_vector = np.where(crossover_mask, mutant, target)
        return trial_vector

    def _stochastic_local_search(self, solution, func, max_iters):
        best_solution = solution.copy()
        best_score = func(best_solution)
        for _ in range(max_iters):
            perturbation = np.random.normal(0.0, 0.1, self.dim)  # Adjusted perturbation
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
                trial = self._crossover(target, mutant, i)
                trial_score = func(trial)
                evaluations += 1

                if trial_score < population_scores[i]:
                    self.population[i] = trial
                    population_scores[i] = trial_score

                    if np.random.rand() < 0.5:  # Simplified hybridization probability
                        improved_solution = self._stochastic_local_search(trial, func, self.dynamic_local_search_iters[i])
                        improved_score = func(improved_solution)
                        evaluations += 1

                        if improved_score < trial_score:
                            self.population[i] = improved_solution
                            population_scores[i] = improved_score

        best_index = np.argmin(population_scores)
        return self.population[best_index]