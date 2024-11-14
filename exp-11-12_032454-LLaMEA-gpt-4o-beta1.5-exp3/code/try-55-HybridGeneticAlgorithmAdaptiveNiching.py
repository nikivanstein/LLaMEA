import numpy as np

class HybridGeneticAlgorithmAdaptiveNiching:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = max(5, int(budget / (8 * dim)))
        self.crossover_rate = 0.7
        self.mutation_rate = 0.1
        self.niche_radius = 0.1

    def __call__(self, func):
        def mutate(individual):
            mutation_vector = np.random.uniform(self.lb, self.ub, self.dim)
            mutated = np.where(np.random.rand(self.dim) < self.mutation_rate, mutation_vector, individual)
            return np.clip(mutated, self.lb, self.ub)

        def crossover(parent1, parent2):
            mask = np.random.rand(self.dim) < self.crossover_rate
            offspring = np.where(mask, parent1, parent2)
            return offspring
        
        def niching_selection(pop, fit):
            selected = []
            fit_sorted_indices = np.argsort(fit)
            for idx in fit_sorted_indices:
                if all(np.linalg.norm(pop[idx] - pop[s]) > self.niche_radius for s in selected):
                    selected.append(idx)
                    if len(selected) > self.population_size / 2:
                        break
            return selected
        
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.population_size
        
        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]
        best_fitness = fitness[best_idx]
        
        while num_evaluations < self.budget:
            selected_indices = niching_selection(population, fitness)
            next_generation = []
            for idx in selected_indices:
                if num_evaluations >= self.budget:
                    break
                partner_idx = np.random.choice(selected_indices)
                offspring = crossover(population[idx], population[partner_idx])
                offspring = mutate(offspring)
                offspring_fitness = func(offspring)
                num_evaluations += 1
                next_generation.append((offspring, offspring_fitness))
            
            next_generation.sort(key=lambda x: x[1])
            for i, (offspring, offspring_fitness) in enumerate(next_generation):
                if i < len(selected_indices):
                    population[selected_indices[i]] = offspring
                    fitness[selected_indices[i]] = offspring_fitness
                    if offspring_fitness < best_fitness:
                        best_individual = offspring
                        best_fitness = offspring_fitness
        
        return best_individual, best_fitness