import sys
from typing import List

ALGO_RECURSIVE = True


def parse_input(input_value: str) -> list:
    """Parse input string of HMM1 from kattis.

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


def transpose(matrix: List[List]) -> List[List]:
    return list(map(list, zip(*matrix)))


def forward_algorithm(A: List[List], B: List[List], pi: List[List], O: List) -> float:

    A_t = transpose(A)
    B_t = transpose(B)

    for idx, emission in enumerate(O):
        if idx == 0:
            alpha = element_wise_multiplication(pi[0], B_t[emission])
        else:
            temp = matrix_mulitplication(A_t, transpose([alpha]))
            alpha = element_wise_multiplication(transpose(temp)[0], B_t[emission])

    return sum(alpha)


def forward_alogortihm_recursive(
    A: List[List], B: List[List], pi: List[List], O: List, alpha: List = None
) -> None:
    if len(O) > 0:
        if not alpha:
            return forward_alogortihm_recursive(
                A, B, pi, O, element_wise_multiplication(pi[0], B[O.pop(0)])
            )
        else:
            if len(O) > 0:
                temp = matrix_mulitplication(A, transpose([alpha]))
                alpha = element_wise_multiplication(transpose(temp)[0], B[O.pop(0)])
                return forward_alogortihm_recursive(A, B, pi, O, alpha)
    else:
        return sum(alpha)


def main():
    file_content = "".join([text for text in sys.stdin])
    A, B, pi, O = parse_input(file_content)

    if ALGO_RECURSIVE:
        A_t = transpose(A)
        B_t = transpose(B)
        print(forward_alogortihm_recursive(A_t, B_t, pi, O))
    else:
        print(forward_algorithm(A, B, pi, O))


if __name__ == "__main__":
    main()
