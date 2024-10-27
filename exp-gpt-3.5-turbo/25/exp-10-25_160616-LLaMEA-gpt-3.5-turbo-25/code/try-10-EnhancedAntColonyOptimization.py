import numpy as np

class EnhancedAntColonyOptimization(AntColonyOptimization):
    def __call__(self, func):
        def update_pheromones(colony, pheromones):
            pheromones *= self.evaporation_rate
            fitness_values = [1.0 / (1.0 + evaluate_solution(ant)) for ant in colony]
            max_fitness = max(fitness_values)

            for ant, fitness in zip(colony, fitness_values):
                pheromones += fitness / max_fitness

            return pheromones

        pheromones = np.ones(self.dim)
        best_solution = None
        best_fitness = np.inf

        colony = initialize_colony()
        for _ in range(self.budget // self.colony_size):
            fitness_values = [evaluate_solution(ant) for ant in colony]
            best_ant_idx = np.argmin(fitness_values)

            if fitness_values[best_ant_idx] < best_fitness:
                best_solution = colony[best_ant_idx]
                best_fitness = fitness_values[best_ant_idx]

            pheromones = update_pheromones(colony, pheromones)
            selection_probs = pheromones / np.sum(pheromones)
            selected_indices = np.random.choice(np.arange(self.colony_size), size=self.colony_size, p=selection_probs)
            colony = colony[selected_indices]

        return best_solution