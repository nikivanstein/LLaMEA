import numpy as np

class EnhancedHybridDE_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30  # Increased population size for higher diversity
        self.population = np.random.uniform(-5.0, 5.0, (self.pop_size, dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.visited_points = 0

    def __call__(self, func):
        F_base = 0.7  # Altered base mutation factor to intensify exploration
        CR_min = 0.65  # Lowered minimum crossover rate for wider exploration
        CR_max = 0.9  # Adjusted maximum crossover rate for enhanced exploitation
        temp = 1.5  # Increased initial temperature for Simulated Annealing

        while self.visited_points < self.budget:
            new_population = np.copy(self.population)
            for i in range(self.pop_size):
                if self.visited_points >= self.budget:
                    break

                # Dynamic Crossover rate based on progress
                CR = CR_min + (CR_max - CR_min) * (1 - (self.visited_points / self.budget))

                # Differential Evolution Step
                indices = np.random.choice(self.pop_size, 3, replace=False)
                while i in indices:
                    indices = np.random.choice(self.pop_size, 3, replace=False)
                a, b, c = self.population[indices]
                mutant = np.clip(a + F_base * (b - c), -5.0, 5.0)

                # Crossover
                crossover_vector = np.copy(self.population[i])
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < CR or j == j_rand:
                        crossover_vector[j] = mutant[j]

                # Selection
                new_fitness = func(crossover_vector)
                self.visited_points += 1

                # Simulated Annealing acceptance criterion
                acceptance_prob = np.exp((self.fitness[i] - new_fitness) / temp)
                if new_fitness < self.fitness[i] or np.random.rand() < acceptance_prob:
                    new_population[i] = crossover_vector
                    self.fitness[i] = new_fitness

            # Adaptive temperature reduction with faster cooling
            temp *= 0.9
            self.population = new_population

        # Return the best-found solution
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]