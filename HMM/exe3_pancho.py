import sys
import itertools
RECURSIVE = True


def elem_wise_product(vector_a, vector_b):
    return [(element_a * element_b) for element_a , element_b in zip(vector_a, vector_b)]

def elem_wise_product_matrix(matrix_a, matrix_b):
    matrix_a_n = len(matrix_a[0])
    flat_a = ([i for row in matrix_a for i in row])
    flat_b = ([i for row in matrix_b for i in row])
    result = elem_wise_product(flat_a, flat_b)

    return [result[i : i + matrix_a_n] for i in range(0, len(result), matrix_a_n)] 


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


def foward_algorithm_recursive (A:list, B:list, pi:list, alpha:list, O:list, first_interation:bool = True) -> float:
    if len(O) > 0:
        if first_interation:
            alpha = elem_wise_product(pi[0], T(B)[O.pop(0)])
            return foward_algorithm_recursive(A, B, pi, alpha, O, False)
        else:
            alpha = elem_wise_product( matrix_mulitplication([alpha] , A)[0], T(B)[O.pop(0)])
            return foward_algorithm_recursive(A, B, pi, alpha, O, False)
    print(sum(alpha))

def foward_algorithm (A:list, B:list, pi:list, O:list) -> float:
    for index, emission in enumerate(O):
        if index == 0:
            alpha = elem_wise_product(pi[0], T(B)[emission])
        else:
            alpha =elem_wise_product( matrix_mulitplication([alpha], A)[0], T(B)[emission])
    print(sum(alpha))

def viterbi_algorithm(A:list, B:list, pi:list, O:list) -> float:
    theta_list = []
    theta_idx_list = []
    for index, emission in enumerate(O):
        if index == 0:
            theta_i = elem_wise_product(pi[0], T(B)[emission])
        else:
            theta_m = [theta_i for _ in theta_i]
            B_m_t = T([T(B)[emission] for _ in T(B)[emission]])

            result =  elem_wise_product_matrix(elem_wise_product_matrix(T(A),theta_m), B_m_t)
            #find the max
            theta_i = ([max(row) for row in result])
            theta_idx = [row.index( max(row)) for row in result]

            theta_list.append(theta_i)
            theta_idx_list.append(theta_idx)

    return theta_list, theta_idx_list


def optimal_sequence(theta_list, theta_idx_list, sequence = None):
    ##get the max
    if len(theta_idx_list) > 0:
        if not sequence:
            index_max_value = theta_list[-1].index( max(theta_list[-1]) )
            sequence = [index_max_value]
            return optimal_sequence(theta_list, theta_idx_list, sequence)
        else:
            # print(sequence)
            next_idx = theta_idx_list.pop()[sequence[-1]]
            sequence.append(next_idx)
            return optimal_sequence(theta_list, theta_idx_list, sequence)
    else:
        return_sequence = [str(i) for i in sequence[::-1]]
        return_sequence = " ".join(return_sequence)
        return return_sequence
    
    

def main():
    file_content = "".join([text for text in sys.stdin])
    A, B, pi, O = parse_input(file_content)
    theta_list, theta_idx_list = viterbi_algorithm(A, B, pi, O)
    # print(theta_list, theta_idx_list)
    print(optimal_sequence(theta_list, theta_idx_list))


if __name__ == "__main__":
    main()
