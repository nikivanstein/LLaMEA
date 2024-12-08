import numpy as np

class ProbabilisticMutation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.mutation_rate = 0.3

    def __call__(self, func):
        population = []
        for _ in range(self.budget):
            x = np.random.uniform(-5.0, 5.0, self.dim)
            population.append(x)
        
        scores = [func(x) for x in population]
        best_idx = np.argmax(scores)
        best_x = population[best_idx]
        best_score = scores[best_idx]

        new_population = []
        for _ in range(self.budget):
            if np.random.rand() < self.mutation_rate:
                mutation = np.random.uniform(-1.0, 1.0, self.dim)
                x = best_x + mutation
            else:
                x = best_x
            new_population.append(x)
        
        scores = [func(x) for x in new_population]
        new_best_idx = np.argmax(scores)
        new_best_x = new_population[new_best_idx]
        new_best_score = scores[new_best_idx]

        if new_best_score > best_score:
            best_x = new_best_x
            best_score = new_best_score

        return best_x, best_score

# Example usage
if __name__ == "__main__":
    budget = 100
    dim = 5
    func = lambda x: x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2
    mutation = ProbabilisticMutation(budget, dim)
    best_x, best_score = mutation(func)
    print("Best x:", best_x)
    print("Best score:", best_score)