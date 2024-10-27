class EnhancedDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30 * dim
        self.mutation_rate = 0.5
        self.crossover_rate = 0.9
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def _initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))

    def _get_fitness(self, population, func):
        return np.array([func(individual) for individual in population])

    def _mutation(self, population, target_idx):
        candidates = population[np.arange(len(population)) != target_idx]
        a, b, c = np.random.choice(len(population), 3, replace=False)
        mutant = np.clip(population[a] + self.mutation_rate * (population[b] - population[c]), self.lower_bound, self.upper_bound)
        gaussian_perturbation = np.random.normal(0, 1, self.dim)
        return mutant + 0.1 * gaussian_perturbation

    def _crossover(self, target_vector, mutant_vector):
        crossover_mask = np.random.rand(self.dim) < self.crossover_rate
        trial_vector = np.where(crossover_mask, mutant_vector, target_vector)
        return trial_vector

    def __call__(self, func):
        population = self._initialize_population()
        evals = 0

        while evals < self.budget:
            for i in range(self.pop_size):
                target_vector = population[i]
                mutant_vector = self._mutation(population, i)
                trial_vector = self._crossover(target_vector, mutant_vector)

                if func(trial_vector) < func(target_vector):
                    population[i] = trial_vector

                evals += 1

                if evals >= self.budget:
                    break

        best_solution = population[np.argmin(self._get_fitness(population, func))]
        return best_solution