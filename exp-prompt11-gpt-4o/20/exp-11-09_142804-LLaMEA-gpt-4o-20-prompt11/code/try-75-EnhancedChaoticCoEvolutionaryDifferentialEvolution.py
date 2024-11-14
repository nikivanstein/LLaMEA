import numpy as np

class EnhancedChaoticCoEvolutionaryDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.mutation_factor = 0.8  # Slightly adjusted mutation factor
        self.population_scaling = 0.5 + (0.4 * np.random.rand())  # Adjusted for more diversity
        self.population_size = int(self.budget / (5 * dim))  # Adjusted size for faster convergence
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.adaptive_crossover_rate = np.linspace(0.5, 0.9, self.population_size)  # Broadened range for exploration
        self.dynamic_mutation_factor = self._init_chaotic_map(self.population_size)
        self.dynamic_local_search_iters = np.random.randint(2, 5, self.population_size)  # Adjusted iterations
        self.subgroup_cooperation_factor = 0.3  # Cooperation factor for co-evolution

    def _init_chaotic_map(self, size):
        return np.sin(np.linspace(0, np.pi, size))  # Chaotic map initialization for better exploration

    def _evaluate_population(self, func):
        return np.array([func(ind) for ind in self.population])

    def _mutate(self, target_idx):
        indices = [idx for idx in range(self.population_size) if idx != target_idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        learning_rate = self.dynamic_mutation_factor[target_idx]
        adaptive_mutation_factor = self.mutation_factor * (1 + np.random.uniform(-0.1, 0.1)) * learning_rate
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

    def _stochastic_local_search(self, solution, func, max_iters):
        best_solution = solution.copy()
        best_score = func(best_solution)
        for _ in range(max_iters):
            perturbation = np.random.normal(0.0, 0.15, self.dim)
            candidate_solution = np.clip(solution + perturbation, self.lower_bound, self.upper_bound)
            candidate_score = func(candidate_solution)
            if candidate_score < best_score:
                best_solution = candidate_solution
                best_score = candidate_score
        return best_solution

    def _adaptive_opposition_based_learning(self):
        mid_point = (self.lower_bound + self.upper_bound) / 2
        opposite_population = mid_point + (mid_point - self.population)
        opposite_population = np.clip(opposite_population, self.lower_bound, self.upper_bound)
        return opposite_population

    def _cooperative_co_evolution(self, func):
        subgroup_size = max(1, self.population_size // 3)
        for i in range(0, self.population_size, subgroup_size):
            subgroup = self.population[i:i + subgroup_size]
            subgroup_scores = [func(ind) for ind in subgroup]
            best_in_subgroup = subgroup[np.argmin(subgroup_scores)]
            self.population[i:i + subgroup_size] = subgroup * (1 - self.subgroup_cooperation_factor) + best_in_subgroup * self.subgroup_cooperation_factor

    def __call__(self, func):
        population_scores = self._evaluate_population(func)
        evaluations = self.population_size

        opposition_population = self._adaptive_opposition_based_learning()
        opposition_scores = np.array([func(ind) for ind in opposition_population])
        evaluations += self.population_size

        for i in range(self.population_size):
            if opposition_scores[i] < population_scores[i]:
                self.population[i] = opposition_population[i]
                population_scores[i] = opposition_scores[i]

        while evaluations < self.budget:
            self._cooperative_co_evolution(func)
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

                    if np.random.rand() < self.dynamic_mutation_factor[i] * 0.65:  # Adjusted for more exploration
                        improved_solution = self._stochastic_local_search(trial, func, self.dynamic_local_search_iters[i])
                        improved_score = func(improved_solution)
                        evaluations += 1

                        if improved_score < trial_score:
                            self.population[i] = improved_solution
                            population_scores[i] = improved_score

        best_index = np.argmin(population_scores)
        return self.population[best_index]