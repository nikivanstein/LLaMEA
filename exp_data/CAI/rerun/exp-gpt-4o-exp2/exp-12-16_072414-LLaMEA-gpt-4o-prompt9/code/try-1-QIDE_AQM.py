import numpy as np

class QIDE_AQM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(100, self.budget // self.dim)
        self.mutation_factor = 0.9  # Updated
        self.crossover_probability = 0.8  # Updated
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.q_probability = 0.1  # Quantum mutation probability

    def initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))

    def mutate(self, population, best_idx):
        idxs = np.random.choice(self.pop_size, 3, replace=False)
        a, b, c = population[idxs]
        mutant = a + self.mutation_factor * (b - c)
        return np.clip(mutant, self.lower_bound, self.upper_bound)

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.crossover_probability
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        offspring = np.where(cross_points, mutant, target)
        return offspring

    def quantum_mutation(self, individual):
        mutation = np.random.uniform(-1.0, 1.0, self.dim)
        quantum_individual = individual + self.q_probability * mutation
        return np.clip(quantum_individual, self.lower_bound, self.upper_bound)

    def adaptive_strategy(self, generation):
        self.mutation_factor = 0.5 + 0.5 * np.cos(np.pi * generation / (self.budget / self.pop_size))
        self.q_probability = 0.1 + 0.1 * np.sin(np.pi * generation / (self.budget / self.pop_size))  # Updated
        
    def __call__(self, func):
        population = self.initialize_population()
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.pop_size
        best_idx = np.argmin(fitness)
        
        generation = 0
        while evaluations < self.budget:
            new_population = np.copy(population)
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break
                
                if np.random.rand() < self.q_probability:  # Quantum-inspired condition
                    mutant = self.quantum_mutation(population[i])
                else:
                    mutant = self.mutate(population, best_idx)
                
                trial = self.crossover(population[i], mutant)
                trial_fitness = func(trial)
                evaluations += 1
                
                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < fitness[best_idx]:
                        best_idx = i

            population = new_population
            generation += 1
            self.adaptive_strategy(generation)
        
        return population[best_idx]