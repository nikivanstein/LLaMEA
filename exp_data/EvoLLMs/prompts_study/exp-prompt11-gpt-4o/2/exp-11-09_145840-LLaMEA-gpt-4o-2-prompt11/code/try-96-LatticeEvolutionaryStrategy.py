import numpy as np

class LatticeEvolutionaryStrategy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 12
        self.mutation_rate = 0.1
        self.step_size = 0.5
        self.evaluated = 0

    def _initialize_population(self):
        return np.clip(np.random.randn(self.population_size, self.dim) + (self.upper_bound + self.lower_bound) / 2, self.lower_bound, self.upper_bound)

    def _mutate(self, individual, best_individual):
        mutation_vector = np.random.standard_cauchy(self.dim) * self.step_size
        guidance_vector = 0.1 * (best_individual - individual)
        return np.clip(individual + mutation_vector + guidance_vector, self.lower_bound, self.upper_bound)

    def _crossover(self, parent1, parent2):
        alpha = np.random.uniform(0, 1, self.dim)
        return np.clip(alpha * parent1 + (1 - alpha) * parent2, self.lower_bound, self.upper_bound)

    def _evaluate_population(self, func, population):
        fitness = np.apply_along_axis(func, 1, population)
        self.evaluated += len(population)
        return fitness

    def _select_best_individual(self, population, fitness):
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]

    def _adaptive_step_size(self, success_rate):
        if success_rate > 0.25:
            self.step_size *= 1.1
        elif success_rate < 0.15:
            self.step_size *= 0.9

    def __call__(self, func):
        assert self.evaluated < self.budget
        
        population = self._initialize_population()
        fitness = self._evaluate_population(func, population)
        best_individual, best_fitness = self._select_best_individual(population, fitness)
        
        while self.evaluated < self.budget:
            new_population = np.array([self._mutate(ind, best_individual) for ind in population])
            
            if np.random.rand() < 0.1:  # Include crossover occasionally
                idx1, idx2 = np.random.choice(self.population_size, 2, replace=False)
                new_population[idx1] = self._crossover(population[idx1], population[idx2])
            
            new_fitness = self._evaluate_population(func, new_population)
            
            success_count = 0
            for i in range(self.population_size):
                if new_fitness[i] < fitness[i]:
                    population[i] = new_population[i]
                    fitness[i] = new_fitness[i]
                    success_count += 1
                    if fitness[i] < best_fitness:
                        best_individual = population[i]
                        best_fitness = fitness[i]

            success_rate = success_count / self.population_size
            self._adaptive_step_size(success_rate)
        
        return best_individual, best_fitness