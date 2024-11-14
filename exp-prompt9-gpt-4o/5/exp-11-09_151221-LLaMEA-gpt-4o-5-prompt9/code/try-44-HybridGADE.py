import numpy as np

class HybridGADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(5 * dim, 20)
        self.mutation_prob = 0.1
        self.crossover_rate = 0.7
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.scaling_factor = 0.8  # Addition for adaptive mutation

    def _initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
    
    def _evaluate_population(self, func, population):
        return np.array([func(ind) for ind in population])
    
    def _select_parents(self, fitness):
        indices = np.arange(self.population_size)
        selected_indices = np.random.choice(indices, 3, replace=False)
        selected_fitness = fitness[selected_indices]
        return selected_indices[np.argmin(selected_fitness)]
    
    def _crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            alpha = np.random.rand()
            return alpha * parent1 + (1 - alpha) * parent2
        return parent1
    
    def _mutate(self, offspring):
        if np.random.rand() < self.mutation_prob:
            mutation_vector = np.random.uniform(-1, 1, self.dim)
            factor = 0.1 + 0.1 * np.random.rand()  # Dynamic adjustment
            return np.clip(offspring + mutation_vector * factor, self.lower_bound, self.upper_bound)
        return offspring

    def _differential_evolution(self, population, idx, best, func):
        a, b, c = population[np.random.choice(self.population_size, 3, replace=False)]
        mutant = np.clip(a + self.scaling_factor * (b - c), self.lower_bound, self.upper_bound)
        trial = self._crossover(population[idx], mutant)
        trial = self._mutate(trial)
        return trial if func(trial) < func(population[idx]) else population[idx]
    
    def __call__(self, func):
        population = self._initialize_population()
        fitness = self._evaluate_population(func, population)
        best_idx = np.argmin(fitness)
        best = population[best_idx]
        eval_count = self.population_size

        while eval_count < self.budget:
            if eval_count % 100 == 0 and eval_count <= self.budget // 2:
                self.population_size = int(self.population_size * 1.1)  # Adaptive population size increase

            new_population = np.copy(population)
            for i in range(self.population_size):
                parent_idx = self._select_parents(fitness)
                parent = population[parent_idx]
                offspring = self._differential_evolution(population, i, best, func)
                new_population[i] = offspring
                eval_count += 1
                if eval_count >= self.budget:
                    break
            
            population = new_population
            fitness = self._evaluate_population(func, population)
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < fitness[best_idx]:
                best_idx = current_best_idx
                best = population[best_idx]
        
        return best