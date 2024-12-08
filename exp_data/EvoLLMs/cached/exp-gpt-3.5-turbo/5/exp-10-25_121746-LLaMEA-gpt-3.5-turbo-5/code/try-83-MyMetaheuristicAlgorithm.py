import numpy as np

class MyMetaheuristicAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.mutation_rate = np.random.uniform(0.2, 0.4)  # Initialize mutation rate

    def __call__(self, func):
        for _ in range(self.budget):
            if np.random.rand() < 0.05:
                self.mutation_rate = np.random.uniform(0.1, 0.5)  # Adjust mutation rate based on probability
            # Implement your optimization strategy here using the updated mutation rate
            pass
        # Return the best solution found
        return best_solution