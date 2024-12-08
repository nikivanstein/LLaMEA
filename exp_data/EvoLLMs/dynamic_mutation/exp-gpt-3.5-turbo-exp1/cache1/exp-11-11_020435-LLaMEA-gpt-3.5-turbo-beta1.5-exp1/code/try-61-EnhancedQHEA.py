import numpy as np

class EnhancedQHEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.num_parents = 4
        self.qubit_num = 2 * dim
        self.search_space = 5.0 * np.random.rand(self.population_size, dim) - 5.0
        self.fitness_values = np.zeros(self.population_size)
    
    def quantum_rotation(self, qubits):
        return qubits / np.linalg.norm(qubits, axis=1)[:, np.newaxis]
    
    def levy_flight(self, size, alpha=1.5, beta=0.5):
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / np.math.gamma((1 + beta) / 2) / 2 ** ((beta - 1) / 2)) ** (1 / beta)
        u = np.random.randn(size) * sigma
        v = np.random.randn(size)
        step = u / abs(v) ** (1 / beta) * alpha
        return step
    
    def opposition_based_learning(self, parents):
        return 2.0 * np.mean(parents, axis=0) - parents
    
    def __call__(self, func):
        for _ in range(self.budget):
            # Evaluate fitness
            for i in range(self.population_size):
                self.fitness_values[i] = func(self.search_space[i])
            
            # Select parents with elitism
            parents_idx = np.argsort(self.fitness_values)[:self.num_parents]
            parents = self.search_space[parents_idx]
            
            # Quantum-inspired rotation
            parents_qubits = np.hstack((parents, np.zeros((self.num_parents, self.dim))))
            rotated_qubits = self.quantum_rotation(parents_qubits)
            children = np.vstack((rotated_qubits[:, :self.dim], -rotated_qubits[:, :self.dim]))
            
            # Levy flight for global search
            step_size = self.levy_flight(self.dim)
            children += step_size
            
            # Opposition-based learning for diversity
            opposite_children = self.opposition_based_learning(children)
            children = np.vstack((children, opposite_children))
            
            # Mutation
            mutation_rate = 0.1
            children += mutation_rate * np.random.randn(4 * self.num_parents, self.dim)
            
            # Update search space with elitism
            combined_population = np.vstack((self.search_space, children))
            all_fitness = np.hstack((self.fitness_values, func(children)))
            top_idx = np.argsort(all_fitness)[:self.population_size]
            self.search_space = combined_population[top_idx]
            self.fitness_values = all_fitness[top_idx]
        return self.search_space[np.argmin(self.fitness_values)]