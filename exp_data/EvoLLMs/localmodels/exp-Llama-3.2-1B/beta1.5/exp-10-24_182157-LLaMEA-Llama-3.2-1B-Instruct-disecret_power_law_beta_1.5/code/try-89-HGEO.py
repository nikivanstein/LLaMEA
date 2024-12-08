import numpy as np

class HGEO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.budgets = np.arange(budget + 1)
        self.grid_size = 10  # hypergrid size
        self.grid = np.zeros((dim, self.grid_size, self.grid_size))
        self.hypergrids = []
        self.population = []  # List to store the selected solution

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

    def select_solution(self):
        # Refine the strategy by changing the number of lines of the selected solution
        self.population.append((np.random.randint(0, self.budget), np.random.randint(0, self.dim), np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size)))

    def mutate(self, individual):
        # Randomly change two lines of the selected solution
        if len(self.population) > 0:
            idx1, idx2 = np.random.choice(len(self.population), 2, replace=False)
            new_individual = self.population[idx1][0] + np.random.randint(-5, 5, self.dim)
            new_individual[idx1] = self.population[idx2][1] + np.random.randint(-5, 5, self.dim)
            new_individual[idx2] = self.population[idx1][2] + np.random.randint(-5, 5, self.dim)
            self.population[idx1] = new_individual
        return individual

    def evolve(self, num_evaluations):
        # Select a random solution and mutate it
        selected_solution = np.random.choice(len(self.population), 1, replace=False)[0]
        selected_solution = self.population[selected_solution]
        
        for _ in range(num_evaluations):
            selected_solution = self.mutate(selected_solution)
        
        # Evaluate the function at the last solution
        x = np.random.rand(self.dim)
        self.grid[:, :, -1] = x
        y = np.array([f(x) for x in np.random.rand(self.dim)])
        
        # Return the optimal solution
        return np.argmax(y)

# Example usage:
if __name__ == "__main__":
    # Define a black box function
    def func(x):
        return x[0]**2 + x[1]**2
    
    # Create an HGEO instance with 10 budget evaluations and 3 dimensions
    hgeo = HGEO(10, 3)
    
    # Optimize the function using HGEO
    optimal_x = hgeo(func)
    print("Optimal solution:", optimal_x)
    
    # Evolve the function for 10 evaluations
    hgeo.evolve(10)
    print("Optimal solution after evolution:", hgeo.func(np.random.rand(3)))