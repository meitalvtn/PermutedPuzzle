# CNN Permutation Analysis Library

## 📜 Philosophy & Purpose

I've structured this repository to serve as a **professional-grade library** for your M.Sc. research project. 
Think of this not as a collection of scripts, but as a robust toolkit. 
Our goal is to build a reliable, modular, and extensible codebase that allows you to design and execute complex experiments with confidence and minimal friction.

As my mentor, YOU'll be guiding me to adhere to the highest standards of software engineering and data science. 
Every function you write, every module you create, must be crafted with the assumption that it will be a core, reusable component of this library. 
This discipline is what separates a student project from professional, impactful research.

---

## 🏛️ Core Architectural Principles

For this library to be successful, we will adhere strictly to the following principles. This is non-negotiable.

* **Modularity and Single Responsibility**: Every function and module must do one thing and do it well. For example, the code that tiles an image should be separate from the code that permutes the tiles, and both should be separate from the data loading logic. This allows us to combine them in flexible ways.
* **Configuration-Driven Experiments**: We will **never** hardcode parameters like learning rates, tile resolutions, or model names into the training scripts. All experiments must be defined and driven by external configuration files (e.g., YAML or JSON) or command-line arguments. This ensures perfect reproducibility and makes it trivial to launch new experimental variations.
* **Robustness and Predictability**: Your code must be resilient. Functions should be pure (i.e., have no side effects) whenever possible. We need to trust that calling `create_permutation(n=9, seed=42)` will return the exact same permutation every single time. This is the foundation of reliable science.
* **Strict Documentation and Typing**: Code is read far more often than it is written. Every function must have a clear docstring explaining what it does, its parameters, and what it returns. Use Python's type hints (`str`, `int`, `np.ndarray`) for all function signatures. This makes the library self-documenting and easier to debug.
