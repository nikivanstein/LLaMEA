import numpy as np

class EnhancedDynamicDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.crossover_rate = 0.85  # Slightly adjusted crossover rate
        self.mutation_factor = 0.7  # Adjusted mutation factor for better exploration
        self.population_scaling = 0.4 + (0.3 * np.random.rand())  # Refined dynamic scaling
        self.dynamic_population_size = int(self.budget / (5 * dim))  # Dynamic population size for enhanced convergence
        self.initial_population_size = self.dynamic_population_size // 2
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.initial_population_size, dim))
        self.max_local_search_iterations = 4  # Increased local search iterations for deeper exploitation
        self.local_search_probability = 0.5  # Increased local search probability

    def _evaluate_population(self, func):
        return np.array([func(ind) for ind in self.population])

    def _mutate(self, target_idx):
        indices = [idx for idx in range(len(self.population)) if idx != target_idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        adaptive_mutation_factor = self.mutation_factor * (1 + np.random.uniform(-0.1, 0.1))
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

    def _expand_population(self, evaluations, func):
        if len(self.population) < self.dynamic_population_size:
            new_individuals_count = min(
                self.dynamic_population_size - len(self.population),
                self.budget - evaluations
            )
            new_individuals = np.random.uniform(
                self.lower_bound, self.upper_bound, (new_individuals_count, self.dim)
            )
            self.population = np.vstack((self.population, new_individuals))
            new_scores = self._evaluate_population(func)[-new_individuals_count:]
            return new_scores, new_individuals_count
        return np.array([]), 0

    def __call__(self, func):
        population_scores = self._evaluate_population(func)
        evaluations = len(self.population)

        while evaluations < self.budget:
            for i in range(len(self.population)):
                if evaluations >= self.budget:
                    break

                target = self.population[i]
                mutant = self._mutate(i)
                trial = self._crossover(target, mutant)
                trial_score = func(trial)
                evaluations += 1

                if trial_score < population_scores[i]:
                    self.population[i] = trial
                    population_scores[i] = trial_score

                    if np.random.rand() < self.local_search_probability:
                        improved_solution = self._stochastic_local_search(trial, func, self.max_local_search_iterations)
                        improved_score = func(improved_solution)
                        evaluations += 1

                        if improved_score < trial_score:
                            self.population[i] = improved_solution
                            population_scores[i] = improved_score

            new_scores, new_evaluations = self._expand_population(evaluations, func)
            evaluations += new_evaluations
            population_scores = np.concatenate((population_scores, new_scores))

        best_index = np.argmin(population_scores)
        return self.population[best_index]