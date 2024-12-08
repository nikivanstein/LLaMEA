import numpy as np

class RefinedQuantumInspiredHarmonySearch(QuantumInspiredHarmonySearch):
    def update_harmony_memory(self, harmony_memory, new_solution):
        min_idx = np.argmin(func(harmony_memory))
        if func(new_solution) < func(harmony_memory[min_idx]):
            harmony_memory[min_idx] = new_solution
        else:
            harmony_memory[np.argmax(func(harmony_memory))] = new_solution