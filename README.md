# MPI Parallel Vector Statistics

A parallel computing assignment implementing statistical calculations on a vector using **MPI (Message Passing Interface)** in C.

## 📌 Overview

This program processes an input vector $X$ of size $n$ using $p$ processors. It distributes the computational load evenly among the available processors (handling cases where $n \% p \neq 0$) and calculates the following statistics:

1.  **Mean Value ($\mu$):** The average of all elements.
2.  **Maximum Value ($m$):** The largest element in the vector.
3.  **Variance ($var$):** The measure of dispersion of the vector elements.
4.  **Delta Vector ($\delta$):** A new vector where each element is the squared difference between the original element and the maximum value $m$.

## ⚙️ Technical Constraints & Implementation

[cite_start]This implementation strictly adheres to the following assignment constraints[cite: 5, 12, 14]:
* **Point-to-Point Communication Only:** Collective functions (like `MPI_Scatter`, `MPI_Reduce`, `MPI_Bcast`) are **not** used. Instead, manual implementations using `MPI_Send` and `MPI_Recv` are employed.
* [cite_start]**Memory Management:** Each processor only allocates memory for the portion of data it processes (Local $X$, Local $\delta$)[cite: 16]. [cite_start]The Master node (Rank 0) handles I/O and final result aggregation[cite: 18].
* [cite_start]**Load Balancing:** The code dynamically calculates the load distribution to handle cases where the vector size $n$ is not perfectly divisible by the number of processors $p$[cite: 20, 63].
* [cite_start]**Menu System:** The program runs in a loop, offering a menu to Continue or Exit[cite: 22].

## 🧮 Formulas Implemented

Given a vector $X = \{x_0, x_1, ..., x_{n-1}\}$:

* **Mean ($\mu$):**
    $$\mu = \frac{1}{n} \sum_{i=0}^{n-1} x_i$$

* **Maximum ($m$):**
    $$m = \max(x_i)$$

* **Variance ($var$):**
    $$var = \frac{1}{n} \sum_{i=0}^{n-1} (x_i - \mu)^2$$

* **Delta Vector ($\delta_i$):**
    $$\delta_i = (x_i - m)^2$$

## 🚀 How to Compile and Run

### Prerequisites
* GCC Compiler
* MPI implementation (e.g., MPICH or OpenMPI)

### Compilation
Open your terminal and compile the code using `mpicc`:

```bash
mpicc main.c -o mpi_stats -lm
