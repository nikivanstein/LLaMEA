import numpy as np

class AQIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = max(5, 3 * dim)
        self.eval_count = 0
        self.alpha = 0.5  # Quantum probability amplitude
        self.mutation_rate = 0.05
        self.adaptation_rate = 0.01

    def initialize_population(self):
        qubits = np.random.uniform(-1, 1, (self.population_size, self.dim))
        return qubits

    def measure_population(self, qubits):
        # Measurement to collapse quantum states to classical solutions
        population = np.sign(qubits) * (self.upper_bound - self.lower_bound) / 2 * qubits + (self.upper_bound + self.lower_bound) / 2
        return np.clip(population, self.lower_bound, self.upper_bound)

    def evaluate_population(self, population, func):
        fitness = np.array([func(ind) for ind in population])
        self.eval_count += len(population)
        return fitness

    def quantum_crossover(self, qubits):
        # Quantum-inspired crossover by random rotation
        indices = np.random.choice(self.population_size, 2, replace=False)
        parent1, parent2 = qubits[indices]
        theta = np.random.uniform(-np.pi/4, np.pi/4, self.dim)
        child1 = np.cos(theta) * parent1 + np.sin(theta) * parent2
        child2 = -np.sin(theta) * parent1 + np.cos(theta) * parent2
        return child1, child2

    def quantum_mutation(self, qubits):
        # Quantum-inspired mutation
        mutation_mask = np.random.rand(*qubits.shape) < self.mutation_rate
        qubits += mutation_mask * np.random.uniform(-self.alpha, self.alpha, qubits.shape)
        return qubits

    def adaptive_parameter_control(self, fitness):
        # Adapt mutation and crossover rates based on diversity
        diversity = np.std(fitness)
        self.mutation_rate = max(0.01, min(0.5, self.mutation_rate + self.adaptation_rate * (0.1 - diversity)))
        self.alpha = max(0.1, min(1.0, self.alpha + self.adaptation_rate * (0.1 - diversity)))

    def __call__(self, func):
        qubits = self.initialize_population()
        population = self.measure_population(qubits)
        fitness = self.evaluate_population(population, func)

        while self.eval_count < self.budget:
            new_qubits = []
            self.adaptive_parameter_control(fitness)
            for _ in range(self.population_size // 2):
                if self.eval_count >= self.budget:
                    break
                child1, child2 = self.quantum_crossover(qubits)
                child1 = self.quantum_mutation(child1)
                child2 = self.quantum_mutation(child2)
                new_qubits.extend([child1, child2])
            
            qubits = np.array(new_qubits[:self.population_size])
            population = self.measure_population(qubits)
            fitness = self.evaluate_population(population, func)

        return population[np.argmin(fitness)]