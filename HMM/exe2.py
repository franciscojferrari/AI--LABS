import sys


def elem_wise_product(vector_a, vector_b):
    return [(element_a * element_b) for element_a , element_b in zip(vector_a, vector_b)]

def matrix_mulitplication(a: list, b: list) -> list:
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

def parse_input(input_value: str) -> list:
    """Parse input string of HMM0 from kattis

    Args:
        input_value (str): input string

    Returns:
        list: A list with the matrixes [A, B, pi]
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

def T(matrix:list)->list:
    return list(map(list, zip(*matrix)))

def foward_algorithm (A, B, pi, O, first_interation = True):
    if len(O) > 0:
        if first_interation:
            temp = [elem_wise_product(pi[0], T(B)[O[0]])]
            O.pop(0)
            return foward_algorithm(A, B, temp, O, False)
        else:
            temp = [elem_wise_product( matrix_mulitplication(pi , A)[0], T(B)[O[0]])]
            O.pop(0)
            return foward_algorithm(A, B, temp, O, False)
    print(sum(pi[0]))
    
def main():
    file_content = "".join([text for text in sys.stdin])
    A, B, pi, O = parse_input(file_content)
    foward_algorithm(A, B, pi, O)
    # parse_output(matrix_mulitplication(matrix_mulitplication(pi, A), B))


if __name__ == "__main__":
    main()
