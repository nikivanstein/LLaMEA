import numpy as np

class EnhancedHybridGADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.base_crossover_rate = 0.9
        self.mutation_factor = 0.8
        self.tournament_size = 3
        self.survival_pressure = 0.45
        self.population = np.random.uniform(-5, 5, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
        
    def select_parents(self, eval_count):
        dynamic_tournament_size = max(2, int(3 * (1 - eval_count / self.budget)))
        indices = np.random.choice(self.population_size, dynamic_tournament_size, replace=False)
        return indices[np.argmin(self.fitness[indices])]

    def crossover(self, parent1, parent2, eval_count):
        adaptive_crossover_rate = self.base_crossover_rate * (0.5 + 0.5 * (eval_count / self.budget))
        mask = np.random.rand(self.dim) < adaptive_crossover_rate
        offspring = np.where(mask, parent1, parent2)
        return offspring

    def mutate(self, target_idx, best_idx, eval_count):
        adaptive_mutation = self.mutation_factor * (1 - (eval_count / self.budget)**2)
        a, b, c = np.random.choice([i for i in range(self.population_size) if i != target_idx], 3, replace=False)
        beta = np.random.rand() * adaptive_mutation
        mutant = self.population[a] + beta * (self.population[b] - self.population[c] + self.population[best_idx] - self.population[target_idx])
        mutant = np.clip(mutant, -5, 5)
        return mutant

    def stochastic_ranking(self):
        ranks = np.argsort(self.fitness)
        for i in range(self.population_size):
            if np.random.rand() < self.survival_pressure:
                ranks[i] = i
        return ranks

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
            
            best_idx = np.argmin(self.fitness)
            best_individual = self.population[best_idx]
            best_fitness = self.fitness[best_idx]

            for i in range(self.population_size):
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

            self.population = new_population
            self.fitness = new_fitness

            worst_idx = np.argmax(self.fitness)
            self.population[worst_idx] = best_individual
            self.fitness[worst_idx] = best_fitness

            if eval_count % 200 == 0:
                self.mutation_factor = 0.8

            if eval_count % 100 == 0:
                self.population_size = max(20, int(self.population_size * 0.9))

            # Apply stochastic ranking
            ranked_indices = self.stochastic_ranking()
            self.population = self.population[ranked_indices]
            self.fitness = self.fitness[ranked_indices]

        return self.get_best_solution()

    def get_best_solution(self):
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]