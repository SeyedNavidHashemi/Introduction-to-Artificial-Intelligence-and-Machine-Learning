# Genetic Algorithm for Knapsack Optimization

A genetic algorithm implementation in Python for solving a multi-constraint knapsack problem, selecting an optimal combination of snacks while satisfying weight, nutritional value, and variety constraints.

## Overview

This project demonstrates the application of genetic algorithms to solve a constrained optimization problem where we need to:
- Maximize nutritional value while respecting weight limits
- Select a diverse variety of items within specified bounds
- Meet multiple constraints simultaneously

## Problem Description

The algorithm selects from a dataset of 19 different snack items, each with available weight and nutritional value. The solution must satisfy:
1. **Weight constraint**: Total weight ≤ specified maximum
2. **Value constraint**: Total nutritional value ≥ specified minimum
3. **Variety constraint**: Number of different snack types within a specified range

## Approach

### Genetic Algorithm Components

- **Chromosome Representation**: Fixed-length chromosomes where each gene represents the selected quantity of a snack item
- **Initial Population**: Randomly generated population of 50 chromosomes
- **Fitness Function**: Penalty-based scoring system that measures constraint satisfaction
- **Selection**: Weighted random selection based on fitness probabilities
- **Crossover**: N-point crossover with configurable probability to mix genetic material between parents
- **Mutation**: Random adjustments to gene values to escape local optima
- **Termination**: Convergence criterion based on average population fitness

### Key Features

- Efficient constraint satisfaction using penalty terms in fitness evaluation
- Configurable hyperparameters (population size, crossover/mutation rates)
- Automatic solution generation with valid constraint satisfaction
- Visualization of algorithm convergence over iterations

## Technical Implementation

Built using Python with NumPy and Pandas for data manipulation and random population generation.

## Usage

1. Load snack data from `snacks.csv`
2. Specify constraints (max weight, min value, snack type range)
3. Run the genetic algorithm
4. Receive multiple valid solutions satisfying all constraints

## Parameters

The algorithm uses configurable hyperparameters:
- `POP_SIZE = 50`: Population size
- `CROSS_PROB = 0.6`: Crossover probability
- `MUTATION_PROB = 0.2`: Mutation probability
- `CUTOFF_POINT = 20`: Convergence threshold

