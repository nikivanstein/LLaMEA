import numpy as np

class ImprovedQHEA(QHEA):
    def __call__(self, func):
        for _ in range(self.budget):
            # Evaluate fitness
            for i in range(self.population_size):
                self.fitness_values[i] = func(self.search_space[i])
            
            # Select parents with elitism
            parents_idx = np.argsort(self.fitness_values)[:self.num_parents]
            parents = self.search_space[parents_idx]

            # Introducing diversity in parent selection
            random_parents_idx = np.random.choice(self.population_size, self.num_parents, replace=False)
            random_parents = self.search_space[random_parents_idx]

            # Quantum-inspired rotation
            parents_qubits = np.hstack((parents, random_parents, np.zeros((self.num_parents, self.dim))))
            rotated_qubits = self.quantum_rotation(parents_qubits)
            children = np.vstack((rotated_qubits[:, :self.dim], -rotated_qubits[:, :self.dim]))
            
            # Mutation
            mutation_rate = 0.1
            children += mutation_rate * np.random.randn(2 * self.num_parents, self.dim)
            
            # Update search space with elitism
            combined_population = np.vstack((self.search_space, children))
            all_fitness = np.hstack((self.fitness_values, func(children)))
            top_idx = np.argsort(all_fitness)[:self.population_size]
            self.search_space = combined_population[top_idx]
            self.fitness_values = all_fitness[top_idx]
        return self.search_space[np.argmin(self.fitness_values)]