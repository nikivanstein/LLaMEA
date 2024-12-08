import numpy as np

class LineRefinedQuantumInspiredEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def entangle_solutions(spark_a, spark_b):
            return 0.5 * (spark_a + spark_b) + np.random.uniform(-0.5, 0.5, size=(self.dim,))

        best_solution = np.random.uniform(-5.0, 5.0, size=(self.dim,))
        best_fitness = func(best_solution)
        
        for _ in range(self.budget):
            sparks = [np.random.uniform(-5.0, 5.0, size=(self.dim,)) for _ in range(5)]
            for spark in sparks:
                new_spark = entangle_solutions(best_solution, spark)
                new_fitness = func(new_spark)
                if new_fitness < best_fitness:
                    best_solution = new_spark
                    best_fitness = new_fitness

            # Quantum Mutation
            mutation_spark = entangle_solutions(best_solution, sparks[0])
            mutation_fitness = func(mutation_spark)
            if mutation_fitness < best_fitness:
                best_solution = mutation_spark
                best_fitness = mutation_fitness

            # Adaptive Quantum Phase
            phase_shift = np.random.uniform(-np.pi, np.pi, size=(self.dim,))
            phase_spark = best_solution * np.exp(1j * phase_shift)
            phase_fitness = func(np.real(phase_spark))
            if phase_fitness < best_fitness:
                best_solution = np.real(phase_spark)
                best_fitness = phase_fitness

            # Line Refinement with probability 0.35
            if np.random.uniform() < 0.35:
                line_direction = np.random.uniform(-1, 1, size=(self.dim,))
                line_direction /= np.linalg.norm(line_direction)
                line_length = np.random.uniform(0.1, 1.0)
                line_point = best_solution + line_length * line_direction
                line_fitness = func(line_point)
                if line_fitness < best_fitness:
                    best_solution = line_point
                    best_fitness = line_fitness

        return best_solution