import sys
from typing import List


def parse_input(input_value: str) -> list:
    """Parse input string of HMM2 from kattis.

    Args:
        input_value (str): input string

    Returns:
        list: A list with the matrices [A, B, pi, O]
    """
    matrixes = []
    for idx, rows in enumerate(input_value.splitlines()):
        if len(rows) > 2:
            values = rows.strip().split(" ")
            if idx > 2:
                matrixes.append([int(item) for item in values[1:]])
            else:
                dimensions = int(float(values[1]))
                matrix_values = [float(item) for item in values[2:]]

                matrixes.append(
                    [
                        matrix_values[i : i + dimensions]
                        for i in range(0, len(matrix_values), dimensions)
                    ]
                )
    return matrixes


def matrix_mulitplication(a: List[List], b: List[List]) -> List[List]:
    """Matrix multiplication.

    Arguments:
        a: Matrix with dimension n, m.
        b: Matrix with dimension m, k.

    Returns:
        Matrix with shape n, k.
    """
    return [
        [sum(ii * jj for ii, jj in zip(i, j)) for j in list(map(list, zip(*b)))]
        for i in a
    ]


def element_wise_multiplication(vec_a: List, vec_b: List) -> List:
    """Element wise vector multiplication."""
    return [(elem_a * elem_b) for elem_a, elem_b in zip(vec_a, vec_b)]


def element_wise_multiplication_matrices(mat_a: List, mat_b: List) -> List:
    """Element wise vector multiplication."""
    col = len(mat_a[0])
    values = [a * b for row_a, row_b in zip(mat_a, mat_b) for a, b in zip(row_a, row_b)]

    return [values[i : i + col] for i in range(0, len(values), col)]


def transpose(matrix: List[List]) -> List[List]:
    return list(map(list, zip(*matrix)))


def algorithm(A: List[List], B: List[List], pi: List[List], O: List) -> float:

    A_t, B_t = transpose(A), transpose(B)

    theta = []
    theta_idx = []

    for idx, emission in enumerate(O):
        if idx == 0:
            theta_i = element_wise_multiplication(pi[0], B_t[emission])
        else:
            theta_m = [theta_i for _ in theta_i]
            b_m = transpose([B_t[emission] for _ in theta_i])

            res = element_wise_multiplication_matrices(A_t, element_wise_multiplication_matrices(theta_m, b_m))

            theta_i = [max(row) for row in res]
            theta_i_idx = [row.index(max(row)) for row in res]

            theta.append(theta_i)
            theta_idx.append(theta_i_idx)

    return theta, theta_idx


def find_sequence(theta: List[List], theta_idx: List[List], sequence: List = None) -> List:
    """Find the sequence by backtracking recursively."""
    if len(theta) > 0:
        if not sequence:
            sequence = [theta[-1].index(max(theta[-1]))]
            return find_sequence(theta, theta_idx, sequence)
        else:
            idx = sequence[-1]
            sequence.append(theta_idx.pop()[idx])
            theta.pop()
            return find_sequence(theta, theta_idx, sequence)
    else:
        return sequence


def format_sequence(sequence):
    return_seq = [str(i) for i in sequence[::-1]]
    return " ".join(return_seq)


def main():
    file_content = "".join([text for text in sys.stdin])
    A, B, pi, O = parse_input(file_content)
    theta, theta_idx = algorithm(A, B, pi, O)
    print(theta,  theta_idx)
    print(format_sequence(find_sequence(theta, theta_idx)))


if __name__ == "__main__":
    main()
