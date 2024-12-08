import numpy as np

class AdaptiveBBOO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.step_size = 1.0
        self.history = []
        self.current_solution = None

    def __call__(self, func):
        # Evaluate the function with the current solution
        score = func(self.current_solution)
        self.history.append((score, self.step_size))

        # Select the next solution based on the score and step size
        if len(self.history) >= self.budget:
            # Calculate the average step size
            avg_step_size = np.mean([s[1] for s in self.history])

            # Select the next solution with the adaptive step size
            next_solution = func(np.array([self.current_solution[i] + self.step_size * avg_step_size for i in range(self.dim)]))
            self.current_solution = next_solution
        else:
            # If the budget is reached, select a random solution
            next_solution = func(np.random.rand(self.dim) * 10 - 5)
            self.current_solution = next_solution

        return next_solution

    def select_next_solution(self, func):
        # Select the next solution based on the probability of exploration and exploitation
        prob_explore = 0.6
        prob_exploit = 0.4
        next_solution = func(self.current_solution)
        if np.random.rand() < prob_explore:
            return func(self.current_solution)
        else:
            return func(np.random.rand(self.dim) * 10 - 5)