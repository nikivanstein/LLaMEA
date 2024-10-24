import numpy as np

class HGEO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.budgets = np.arange(budget + 1)
        self.grid_size = 10  # hypergrid size
        self.grid = np.zeros((dim, self.grid_size, self.grid_size))
        self.hypergrids = []
        self.population = []

    def __call__(self, func):
        def f(x):
            return np.array(func(x))
        
        # Evaluate the function at each hypergrid location
        for i in range(self.budget):
            x = np.random.rand(self.dim)
            self.grid[:, :, i] = x
            
            # Generate new hypergrids by perturbing the current location
            for j in range(self.grid_size):
                for k in range(self.grid_size):
                    self.grid[:, :, i + 1] = self.grid[:, :, i] + np.random.uniform(-1, 1, self.dim)
            
            # Evaluate the function at each new location
            y = np.array([f(x) for x in np.random.rand(self.dim)])
            
            # Update the hypergrid and its budget
            self.grid[:, :, i + 1] = x
            self.budgets[i + 1] = min(self.budgets[i] + 1, self.budget)
            
            # Add the new hypergrid to the list of hypergrids
            self.hypergrids.append(self.grid[:, :, i + 1])
        
        # Evaluate the function at the last hypergrid location
        x = np.random.rand(self.dim)
        self.grid[:, :, -1] = x
        y = np.array([f(x) for x in np.random.rand(self.dim)])
        
        # Return the optimal solution
        return np.argmax(y)

    def select_solution(self, population):
        # Novel Heuristic Algorithm: "Adaptive Step Size"
        adaptive_step_size = 0.5 * (1 / np.sqrt(len(population)))
        selection_prob = 0.5
        selection_prob *= np.exp(-((np.random.rand(len(population)) / adaptive_step_size) ** 2))
        selection_prob /= np.sum(selection_prob)
        
        # Select the individual with the highest fitness
        selected_individual = np.argmax(population)
        
        # Refine the solution based on the adaptive step size
        if np.random.rand() < selection_prob:
            # Increase the step size
            self.grid[:, :, selected_individual] += np.random.uniform(-adaptive_step_size, adaptive_step_size, self.dim)
            self.population[selected_individual] = f(self.grid[:, :, selected_individual])
        
        # Decrease the step size
        else:
            self.grid[:, :, selected_individual] -= np.random.uniform(-adaptive_step_size, adaptive_step_size, self.dim)
            self.population[selected_individual] = f(self.grid[:, :, selected_individual])
        
        return selected_individual

# Example usage:
if __name__ == "__main__":
    # Define a black box function
    def func(x):
        return x[0]**2 + x[1]**2
    
    # Create an HGEO instance with 10 budget evaluations and 3 dimensions
    hgeo = HGEO(10, 3)
    
    # Optimize the function using HGEO
    population = hgeo(func)
    optimal_x = hgeo(func)(population)
    print("Optimal solution:", optimal_x)