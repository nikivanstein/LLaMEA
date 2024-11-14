import numpy as np

class HybridGADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(5 * dim, 20)
        self.mutation_prob = 0.1  # Adaptive mutation probability
        self.crossover_rate = 0.8  # Adaptive crossover rate
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.elite_fraction = 0.1

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
    
    def _mutate(self, offspring, generation, max_generations):
        adaptive_mutation_prob = self.mutation_prob * (1 - generation / max_generations)
        if np.random.rand() < adaptive_mutation_prob:
            mutation_vector = np.random.uniform(-1, 1, self.dim)
            return np.clip(offspring + mutation_vector * 0.1, self.lower_bound, self.upper_bound)
        return offspring

    def _differential_evolution(self, population, idx, best, func, generation, max_generations):
        a, b, c = population[np.random.choice(self.population_size, 3, replace=False)]
        mutant = np.clip(a + 0.8 * (b - c), self.lower_bound, self.upper_bound)
        trial = self._crossover(population[idx], mutant)
        trial = self._mutate(trial, generation, max_generations)
        return trial if func(trial) < func(population[idx]) else population[idx]
    
    def __call__(self, func):
        population = self._initialize_population()
        fitness = self._evaluate_population(func, population)
        best_idx = np.argmin(fitness)
        best = population[best_idx]
        eval_count = self.population_size
        max_generations = self.budget // self.population_size
        generation = 0

        while eval_count < self.budget:
            new_population = np.copy(population)
            num_elites = int(self.elite_fraction * self.population_size)
            elite_indices = np.argsort(fitness)[:num_elites]
            new_population[:num_elites] = population[elite_indices]

            for i in range(num_elites, self.population_size):
                parent_idx = self._select_parents(fitness)
                parent = population[parent_idx]
                offspring = self._differential_evolution(population, i, best, func, generation, max_generations)
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
            
            generation += 1
        
        return best