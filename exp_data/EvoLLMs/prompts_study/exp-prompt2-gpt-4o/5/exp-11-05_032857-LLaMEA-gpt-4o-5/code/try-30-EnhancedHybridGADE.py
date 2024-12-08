import numpy as np

class EnhancedHybridGADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.base_crossover_rate = 0.9
        self.mutation_factor = 0.8
        self.tournament_size = 3
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
        scaling_factor = np.random.uniform(0.5, 1.0)  # Adaptive mutation scaling
        mutant = mutant + scaling_factor * (self.population[best_idx] - self.population[a])
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

                if offspring_fitness < self.fitness[i]:
                    new_population[i] = offspring
                    new_fitness[i] = offspring_fitness
                else:
                    new_population[i] = self.population[i]
                    new_fitness[i] = self.fitness[i]

                if eval_count >= self.budget:
                    break

            fitness_indices = np.argsort(new_fitness)
            sigma = 0.1
            ranks = np.arange(1, self.population_size + 1)
            probabilities = (1 - sigma) / self.population_size + sigma * (1 - ranks / self.population_size) / ranks.sum()
            selected_indices = np.random.choice(fitness_indices, self.population_size, p=probabilities, replace=False)

            self.population = new_population[selected_indices]
            self.fitness = new_fitness[selected_indices]

        return self.get_best_solution()

    def get_best_solution(self):
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]