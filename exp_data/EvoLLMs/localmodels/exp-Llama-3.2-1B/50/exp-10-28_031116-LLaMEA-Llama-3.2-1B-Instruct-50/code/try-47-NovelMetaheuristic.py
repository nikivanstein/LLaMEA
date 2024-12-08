import numpy as np
import random

class NovelMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = self.generate_population()
        self.best_solution = None
        self.best_score = -np.inf

    def generate_population(self):
        # Initialize population with random solutions
        population = []
        for _ in range(100):
            solution = np.random.uniform(-5.0, 5.0, self.dim)
            population.append(solution)
        return population

    def __call__(self, func):
        # Evaluate the function for each solution in the population
        for solution in self.population:
            score = func(solution)
            # If the score is better than the current best score, update the best solution and score
            if score > self.best_score:
                self.best_solution = solution
                self.best_score = score
                # Change the individual lines of the selected solution to refine its strategy
                if random.random() < 0.45:
                    solution[0] += 0.1
                    solution[1] += 0.2
                elif random.random() < 0.3:
                    solution[0] -= 0.1
                    solution[1] -= 0.2
                elif random.random() < 0.2:
                    solution[0] += 0.3
                    solution[1] += 0.4
                elif random.random() < 0.15:
                    solution[0] -= 0.3
                    solution[1] -= 0.4
        return self.best_solution

# BBOB test suite functions
def func1(solution):
    return np.sum(solution ** 2)

def func2(solution):
    return np.abs(solution)

def func3(solution):
    return np.prod(solution)

def func4(solution):
    return np.mean(solution)

# Run the optimization algorithm
opt = NovelMetaheuristic(100, 10)
func = random.choice([func1, func2, func3, func4])
best_solution = opt(func)
print("Best solution:", best_solution)
print("Best score:", best_solution[0])
print("Score:", best_solution[1])