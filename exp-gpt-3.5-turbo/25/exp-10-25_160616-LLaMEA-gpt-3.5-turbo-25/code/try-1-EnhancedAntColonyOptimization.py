import numpy as np

class EnhancedAntColonyOptimization(AntColonyOptimization):
    def __call__(self, func):
        def roulette_wheel_selection(colony, pheromones):
            total_pheromones = np.sum(pheromones)
            probabilities = pheromones / total_pheromones
            selected_index = np.random.choice(np.arange(self.colony_size), p=probabilities)
            return colony[selected_index]

        # Initialize pheromones
        pheromones = np.ones(self.dim)

        # Initialize the best solution and its fitness value
        best_solution = None
        best_fitness = np.inf

        colony = initialize_colony()
        for _ in range(self.budget // self.colony_size):
            for _ in range(self.colony_size):
                ant_solution = roulette_wheel_selection(colony, pheromones)
                fitness = evaluate_solution(ant_solution)
                if fitness < best_fitness:
                    best_solution = ant_solution
                    best_fitness = fitness

                pheromones = update_pheromones([ant_solution], pheromones)

            colony = np.array([roulette_wheel_selection(colony, pheromones) for _ in range(self.colony_size)])

        return best_solution