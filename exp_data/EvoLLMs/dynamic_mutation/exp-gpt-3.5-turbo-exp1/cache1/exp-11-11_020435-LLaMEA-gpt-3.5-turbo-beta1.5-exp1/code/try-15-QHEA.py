import numpy as np

class QHEA:
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
    
    def __call__(self, func):
        for _ in range(self.budget):
            # Evaluate fitness
            for i in range(self.population_size):
                self.fitness_values[i] = func(self.search_space[i])
            
            # Select parents
            parents_idx = np.argsort(self.fitness_values)[:self.num_parents]
            parents = self.search_space[parents_idx]
            
            # Quantum-inspired rotation
            parents_qubits = np.hstack((parents, np.zeros((self.num_parents, self.dim))))
            rotated_qubits = self.quantum_rotation(parents_qubits)
            children = np.vstack((rotated_qubits[:, :self.dim], -rotated_qubits[:, :self.dim]))
            
            # Mutation
            mutation_rate = 0.1
            children += mutation_rate * np.random.randn(2 * self.num_parents, self.dim)
            
            # Update search space
            self.search_space[:self.num_parents] = parents
            self.search_space[self.num_parents:] = children
        return self.search_space[np.argmin(self.fitness_values)]