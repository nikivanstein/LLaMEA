import numpy as np

class EnhancedHybridGADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.base_crossover_rate = 0.9
        self.mutation_factor = 0.8
        self.tournament_size = 3
        self.reinit_probability = 0.05  # New parameter for reinitialization probability
        self.population = np.random.uniform(-5, 5, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
        
    def select_parents(self):
        indices = np.random.choice(self.population_size, self.tournament_size, replace=False)
        return indices[np.argmin(self.fitness[indices])]

    def crossover(self, parent1, parent2, eval_count):
        adaptive_crossover_rate = self.base_crossover_rate * (0.5 + 0.5 * (eval_count / self.budget))
        mask = np.random.rand(self.dim) < adaptive_crossover_rate
        offspring = np.where(mask, parent1, parent2)
        return offspring

    def mutate(self, target_idx, best_idx, eval_count):
        adaptive_mutation = self.mutation_factor * (1 - eval_count / self.budget)
        a, b, c = np.random.choice([i for i in range(self.population_size) if i != target_idx], 3, replace=False)
        mutant = self.population[a] + adaptive_mutation * (self.population[b] - self.population[c])
        mutant = np.clip(mutant, -5, 5)
        return mutant

    def __call__(self, func):
        eval_count = 0

        for i in range(self.population_size):
            self.fitness[i] = func(self.population[i])
            eval_count += 1
            if eval_count >= self.budget:
                return self.get_best_solution()

        while eval_count < self.budget:
            new_population = np.empty_like(self.population)
            new_fitness = np.empty_like(self.fitness)
            for i in range(self.population_size):
                best_idx = np.argmin(self.fitness)
                mutant = self.mutate(i, best_idx, eval_count)
                offspring = self.crossover(self.population[i], mutant, eval_count)

                offspring_fitness = func(offspring)
                eval_count += 1

                # Select the better one between parent and offspring
                if offspring_fitness < self.fitness[i]:
                    new_population[i] = offspring
                    new_fitness[i] = offspring_fitness
                else:
                    new_population[i] = self.population[i]
                    new_fitness[i] = self.fitness[i]
                
                # Reinitialize underperforming individuals
                if np.random.rand() < self.reinit_probability:
                    new_population[i] = np.random.uniform(-5, 5, self.dim)
                    new_fitness[i] = func(new_population[i])
                    eval_count += 1

                if eval_count >= self.budget:
                    break

            self.population = new_population
            self.fitness = new_fitness

        return self.get_best_solution()

    def get_best_solution(self):
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]