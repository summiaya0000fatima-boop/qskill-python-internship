"""
QSkill Internship - Task 3 (Slab 1, Beginner)
Matrix Operations Tool using Python and NumPy.
Supports: addition, subtraction, multiplication, transpose, and determinant.

Run interactively (python task3_matrix_tool.py) or import and use
MatrixTool programmatically / see the __main__ demo block for a
non-interactive example run.
"""

import numpy as np


class MatrixTool:
    """A small interactive matrix-operations utility built on NumPy."""

    @staticmethod
    def input_matrix(name: str) -> np.ndarray:
        """Prompt the user to type in a matrix, row by row."""
        print(f"\nEnter matrix {name}:")
        rows = int(input(f"  Number of rows for {name}: "))
        cols = int(input(f"  Number of columns for {name}: "))
        print(f"  Enter each row of {name} as space-separated numbers.")

        data = []
        for r in range(rows):
            while True:
                row_str = input(f"  Row {r + 1}: ").strip().split()
                if len(row_str) != cols:
                    print(f"  Expected {cols} values, got {len(row_str)}. Try again.")
                    continue
                data.append([float(x) for x in row_str])
                break
        return np.array(data)

    @staticmethod
    def display(label: str, matrix: np.ndarray):
        print(f"\n{label}:")
        print(matrix)

    @staticmethod
    def add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if a.shape != b.shape:
            raise ValueError(f"Shape mismatch for addition: {a.shape} vs {b.shape}")
        return a + b

    @staticmethod
    def subtract(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if a.shape != b.shape:
            raise ValueError(f"Shape mismatch for subtraction: {a.shape} vs {b.shape}")
        return a - b

    @staticmethod
    def multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if a.shape[1] != b.shape[0]:
            raise ValueError(
                f"Cannot multiply {a.shape} by {b.shape}: "
                f"columns of A must equal rows of B"
            )
        return a @ b

    @staticmethod
    def transpose(a: np.ndarray) -> np.ndarray:
        return a.T

    @staticmethod
    def determinant(a: np.ndarray) -> float:
        if a.shape[0] != a.shape[1]:
            raise ValueError(f"Determinant requires a square matrix, got {a.shape}")
        return float(np.linalg.det(a))


MENU = """
=================================================
        MATRIX OPERATIONS TOOL (NumPy)
=================================================
1. Addition (A + B)
2. Subtraction (A - B)
3. Multiplication (A x B)
4. Transpose (A^T)
5. Determinant (|A|)
6. Exit
=================================================
"""


def run_interactive():
    tool = MatrixTool()
    print(MENU)

    while True:
        choice = input("Select an operation (1-6): ").strip()

        if choice == "6":
            print("Goodbye!")
            break

        try:
            if choice in ("1", "2", "3"):
                a = tool.input_matrix("A")
                b = tool.input_matrix("B")
                tool.display("Matrix A", a)
                tool.display("Matrix B", b)

                if choice == "1":
                    result = tool.add(a, b)
                    tool.display("Result: A + B", result)
                elif choice == "2":
                    result = tool.subtract(a, b)
                    tool.display("Result: A - B", result)
                else:
                    result = tool.multiply(a, b)
                    tool.display("Result: A x B", result)

            elif choice == "4":
                a = tool.input_matrix("A")
                tool.display("Matrix A", a)
                tool.display("Result: A^T (Transpose)", tool.transpose(a))

            elif choice == "5":
                a = tool.input_matrix("A")
                tool.display("Matrix A", a)
                det = tool.determinant(a)
                print(f"\nResult: determinant(A) = {det:.4f}")

            else:
                print("Invalid choice. Please select 1-6.")

        except ValueError as e:
            print(f"\nError: {e}")

        print(MENU)


def demo():
    """Non-interactive demo showing every operation with sample matrices."""
    tool = MatrixTool()

    A = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    B = np.array([[7, 8, 9], [10, 11, 12]], dtype=float)
    C = np.array([[2, 0], [1, 3], [4, 1]], dtype=float)
    D = np.array([[4, 7], [2, 6]], dtype=float)

    print(MENU)
    tool.display("Matrix A (2x3)", A)
    tool.display("Matrix B (2x3)", B)
    tool.display("Matrix C (3x2)", C)
    tool.display("Matrix D (2x2, square)", D)

    print("\n--- 1. Addition: A + B ---")
    tool.display("Result", tool.add(A, B))

    print("\n--- 2. Subtraction: A - B ---")
    tool.display("Result", tool.subtract(A, B))

    print("\n--- 3. Multiplication: A x C ---")
    tool.display("Result", tool.multiply(A, C))

    print("\n--- 4. Transpose: A^T ---")
    tool.display("Result", tool.transpose(A))

    print("\n--- 5. Determinant: |D
