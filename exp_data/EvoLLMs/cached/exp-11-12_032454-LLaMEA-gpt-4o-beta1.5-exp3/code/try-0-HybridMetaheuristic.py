import numpy as np

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = max(5, int(budget / (10 * dim)))  # heuristic population size
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.temperature = 1000  # initial temperature for simulated annealing
        self.cooling_rate = 0.95

    def __call__(self, func):
        # Randomly initialize the population
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.population_size
        
        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        best_fitness = fitness[best_index]
        
        while num_evaluations < self.budget:
            for i in range(self.population_size):
                # Differential Evolution Mutation
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                donor_vector = population[a] + self.mutation_factor * (population[b] - population[c])
                donor_vector = np.clip(donor_vector, self.lb, self.ub)
                
                # Crossover
                trial_vector = np.copy(population[i])
                for j in range(self.dim):
                    if np.random.rand() < self.crossover_rate:
                        trial_vector[j] = donor_vector[j]
                
                # Evaluate trial vector
                trial_fitness = func(trial_vector)
                num_evaluations += 1
                
                # Selection
                if trial_fitness < fitness[i]:
                    population[i] = trial_vector
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_solution = trial_vector
                        best_fitness = trial_fitness

                # Simulated Annealing Acceptance
                elif np.random.rand() < np.exp((fitness[i] - trial_fitness) / self.temperature):
                    population[i] = trial_vector
                    fitness[i] = trial_fitness
                
                # Update best solution
                if fitness[i] < best_fitness:
                    best_solution = population[i]
                    best_fitness = fitness[i]
            
            # Cooling
            self.temperature *= self.cooling_rate

            if num_evaluations >= self.budget:
                break

        return best_solution, best_fitness