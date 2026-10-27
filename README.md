# Assignment 2: Vector Embedding for Capability Composition

## Overview

This project implements a baseline vector-embedding system for representing application states, goals, and capabilities. It explores whether numerical feature vectors can help describe functional similarity, capability compatibility, composition, goal relevance, and operational properties.

The example domain is an online shopping application with capabilities such as `CreateOrder`, `MakePayment`, `SendNotification`, `CancelCart`, `CreateOrderDatabase`, and `CheckInventory`.

## Features

- Encodes application states as numerical vectors.
- Encodes goal specifications as numerical vectors.
- Encodes capabilities using effects, inputs, outputs, type, resources, constraints, cost, reliability, and availability.
- Calculates cosine similarity between vector representations.
- Checks direct capability compatibility using preconditions, effects, and inputs/outputs.
- Builds a composite representation from a sequence of capabilities.
- Evaluates capability-to-goal similarity.
- Prints experiment results in the terminal.
- Saves a CSV file and graphs for use in the technical report.

## Requirements

- Python 3.10 or later recommended
- NumPy
- scikit-learn
- Matplotlib

## Installation

Open PowerShell or a terminal in the project folder and install the dependencies:

```powershell
python -m pip install numpy scikit-learn matplotlib
```

If your system uses `python3` instead of `python`, use:

```powershell
python3 -m pip install numpy scikit-learn matplotlib
```

## How to Run

Save the program as `capability_embedding.py`, open a terminal in the same folder, and run:

```powershell
python capability_embedding.py
```

Or:

```powershell
python3 capability_embedding.py
```

The program prints the experiment results in the terminal and generates the output files listed below in the current working directory.

## Experiments

### 1. State and goal encoding

Encodes the example initial application state and desired goal into fixed-length feature vectors.

### 2. Capability compatibility

Tests selected pairs, including:

- `CreateOrder -> MakePayment`
- `CreateOrder -> CancelCart`
- `CreateOrder -> SendNotification`
- `CreateOrderDatabase -> MakePayment`

Compatibility is checked symbolically using capability preconditions, effects, and input/output names.

### 3. Capability similarity

Calculates cosine similarity for selected capability pairs. A higher score means the vectors point in a more similar direction; it is not a probability of compatibility.

### 4. Capability composition

Creates a composite vector for:

`CreateOrder -> MakePayment -> SendNotification`

The program checks the sequence using accumulated effects and outputs from earlier capabilities.

### 5. Goal relevance

Calculates cosine similarity between each capability vector and the goal vector.

### 6. Operational attributes

Displays each capability's cost, reliability, and availability.

## Generated Output Files

When the graph-enabled version of the program runs successfully, it creates:

| File | Description |
|---|---|
| `capability_similarity.png` | Bar chart comparing selected capability-pair cosine similarity scores. |
| `goal_relevance.png` | Bar chart comparing capability-to-goal similarity scores. |
| `operational_attributes.png` | Bar chart comparing capability cost and reliability. |
| `experiment_results.csv` | CSV file containing selected capability-pair similarity scores. |

The terminal also displays compatibility results, composition validity, vector dimensions, goal relevance scores, and operational attributes.

## Project Structure

A simple project folder can look like this:

```text
CapabilityEmbedding/
├── capability_embedding.py
├── README.md
├── capability_similarity.png       # generated after running
├── goal_relevance.png               # generated after running
├── operational_attributes.png       # generated after running
└── experiment_results.csv           # generated after running
```

Generated graph and CSV files appear only after the program runs successfully.

## Method Summary

The implementation uses a fixed, manually designed feature vocabulary to produce interpretable vectors. Cosine similarity compares vector direction. A separate symbolic check evaluates whether a capability's effects and outputs satisfy later capabilities' preconditions and inputs.

This distinction matters: **similarity and composability are different properties**. Two capabilities can be functionally similar without being directly composable.

## Limitations

- The feature vocabulary and feature values are manually designed; the system does not train an embedding model from a large dataset.
- The compatibility checker is simplified and does not fully model external input binding, conflicting effects, state resets, or all domain constraints.
- The composite vector is the arithmetic mean of the individual capability vectors, which is a simple baseline rather than a learned composition function.
- Cosine similarity is a comparison score, not a probability or proof of compatibility.
- The example dataset is small and uses illustrative operational values.

These limitations should be discussed in the technical report.

## Troubleshooting

### `ModuleNotFoundError: No module named 'numpy'`

Run:

```powershell
python -m pip install numpy
```

### `ModuleNotFoundError: No module named 'sklearn'`

Install scikit-learn:

```powershell
python -m pip install scikit-learn
```

### `ModuleNotFoundError: No module named 'matplotlib'`

Install Matplotlib:

```powershell
python -m pip install matplotlib
```

### `SyntaxError: '(' was never closed`

Make sure you are running the corrected, complete Python file rather than an older or partially edited version.

## Assignment Deliverables

This code supports the implementation and experimental-results portions of the assignment. The complete submission should also include:

1. Formal embedding design and mathematical specification.
2. Experimental dataset and definitions.
3. Technical report covering methodology, results, analysis, limitations, and conclusion.
