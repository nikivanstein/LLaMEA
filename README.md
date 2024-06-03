<p align="center">
  <img src="logo.png" alt="LLaMEA Logo" width="200"/>
</p>

<h1 align="center">LLaMEA: Large Language Model Evolutionary Algorithm</h1>

<p align="center">
  <a href="https://pypi.org/project/llamea/">
    <img src="https://badge.fury.io/py/llamea.svg" alt="PyPI version" height="20">
  </a>
</p>

## Introduction

LLaMEA (Large Language Model Evolutionary Algorithm) is an innovative framework that leverages the power of large language models (LLMs) such as GPT-4 for the automated generation and refinement of metaheuristic optimization algorithms. The framework utilizes a novel approach to evolve and optimize algorithms iteratively based on performance metrics and runtime evaluations without requiring extensive prior algorithmic knowledge. This makes LLaMEA an ideal tool for both research and practical applications in fields where optimization is crucial.

## Features

- **Automated Algorithm Generation**: Automatically generates and refines algorithms using GPT models.
- **Performance Evaluation**: Integrates with the IOHexperimenter for real-time performance feedback, guiding the evolutionary process to generate metaheuristic optimization algorithms.
- **Customizable Evolution Strategies**: Supports configuration of evolutionary strategies to explore algorithmic design spaces effectively.
- **Extensible and Modular**: Designed to be flexible, allowing users to integrate other models and evaluation tools.

## Getting Started

### Prerequisites

- Python 3.8 or later
- OpenAI API key for accessing GPT models

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourgithub/llamea.git
   ```
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

### Setting Up Your Environment

1. Set up an OpenAI API key:
   - Obtain an API key from [OpenAI](https://openai.com/).
   - Set the API key in your environment variables:
     ```bash
     export OPENAI_API_KEY='your_api_key_here'
     ```

2. Running an Experiment

    To run an optimization experiment using LLaMEA:

    ```python
    from llamea import LLaMEA

    # Define your evaluation function
    def your_evaluation_function(solution):
        # Implementation of your function
        pass

    # Initialize LLaMEA with your API key and other parameters
    optimizer = LLaMEA(f=your_evaluation_function, api_key="your_api_key_here")

    # Run the optimizer
    best_solution, best_fitness = optimizer.run()
    print(f"Best Solution: {best_solution}, Fitness: {best_fitness}")
    ```

## Contributing

Contributions to LLaMEA are welcome! Here are a few ways you can help:

- **Report Bugs**: Use [GitHub Issues](https://github.com/yourgithub/llamea/issues) to report bugs.
- **Feature Requests**: Suggest new features or improvements.
- **Pull Requests**: Submit PRs for bug fixes or feature additions.

Please refer to CONTRIBUTING.md for more details on contributing guidelines.

## License

Distributed under the MIT License. See `LICENSE` for more information.


## Citation

If you use LLaMEA in your research, please consider citing the associated paper:

```bibtex
@misc{vanstein2024llamea,
      title={LLaMEA: A Large Language Model Evolutionary Algorithm for Automatically Generating Metaheuristics}, 
      author={Niki van Stein and Thomas Bäck},
      year={2024},
      eprint={2405.20132},
      archivePrefix={arXiv},
      primaryClass={cs.NE}
}
```

---

For more details, please refer to the documentation and tutorials available in the repository.
