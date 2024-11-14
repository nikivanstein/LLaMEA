import numpy as np

class QuantumFireflyAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.alpha = 0.5  # attractiveness coefficient base
        self.beta_min = 0.2  # minimum value of attractiveness
        self.gamma = 1.0  # absorption coefficient
        self.population_size = max(10, int(budget / (5 * dim)))  # heuristic for population size
        self.quantum_prob = 0.1  # probability of quantum-inspired move

    def __call__(self, func):
        # Initialize the population
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.population_size
        
        # Main loop
        while num_evaluations < self.budget:
            for i in range(self.population_size):
                for j in range(self.population_size):
                    if fitness[j] < fitness[i]:  # Move i towards j if j is brighter
                        r = np.linalg.norm(population[i] - population[j])
                        beta = self.beta_min + (1 - self.beta_min) * np.exp(-self.gamma * r**2)
                        step = beta * (population[j] - population[i]) + self.alpha * (np.random.rand(self.dim) - 0.5)
                        population[i] += step
                        population[i] = np.clip(population[i], self.lb, self.ub)
                        
                        if np.random.rand() < self.quantum_prob:  # Quantum-inspired move
                            quantum_step = np.random.uniform(self.lb, self.ub, self.dim)
                            population[i] = (population[i] + quantum_step) / 2
                            population[i] = np.clip(population[i], self.lb, self.ub)
                        
                        current_fitness = func(population[i])
                        num_evaluations += 1
                        if current_fitness < fitness[i]:
                            fitness[i] = current_fitness
                        
                        if num_evaluations >= self.budget:
                            break
                if num_evaluations >= self.budget:
                    break
            
        # Return the best solution found
        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]