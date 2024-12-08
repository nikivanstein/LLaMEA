import numpy as np

class AQEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = max(5, 4 * dim)
        self.eval_count = 0
        self.alpha = 0.6  # Probability amplitude
        
    def initialize_population(self):
        # Quantum-inspired initialization (superposition)
        real_part = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        imaginary_part = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        return real_part, imaginary_part

    def evaluate_population(self, population, func):
        fitness = np.array([func(ind) for ind in population])
        self.eval_count += len(population)
        return fitness

    def quantum_crossover(self, real_part, imaginary_part):
        # Quantum-inspired crossover: combining real and imaginary parts
        new_real = self.alpha * real_part + (1 - self.alpha) * imaginary_part
        new_imaginary = self.alpha * imaginary_part + (1 - self.alpha) * real_part
        return new_real, new_imaginary

    def adaptive_mutation(self, real_part, imaginary_part, best_idx):
        # Adaptive mutation based on the best individual
        mutation_rate = max(0.1, 1.0 - (self.eval_count / self.budget))
        best_real = real_part[best_idx]
        best_imaginary = imaginary_part[best_idx]
        real_mutation = np.random.uniform(-mutation_rate, mutation_rate, (self.population_size, self.dim))
        imaginary_mutation = np.random.uniform(-mutation_rate, mutation_rate, (self.population_size, self.dim))
        new_real = np.clip(best_real + real_mutation, self.lower_bound, self.upper_bound)
        new_imaginary = np.clip(best_imaginary + imaginary_mutation, self.lower_bound, self.upper_bound)
        return new_real, new_imaginary

    def __call__(self, func):
        real_part, imaginary_part = self.initialize_population()
        composite_population = real_part + 1j * imaginary_part
        population = np.real(composite_population)
        fitness = self.evaluate_population(population, func)

        while self.eval_count < self.budget:
            best_idx = np.argmin(fitness)
            real_part, imaginary_part = self.quantum_crossover(real_part, imaginary_part)

            # Perform adaptive mutation
            real_part, imaginary_part = self.adaptive_mutation(real_part, imaginary_part, best_idx)
            composite_population = real_part + 1j * imaginary_part
            population = np.real(composite_population)
            
            new_fitness = self.evaluate_population(population, func)
            if np.min(new_fitness) < np.min(fitness):
                fitness = new_fitness
        
        return population[np.argmin(fitness)]