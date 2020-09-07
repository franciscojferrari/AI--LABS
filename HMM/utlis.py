from typing import List, Tuple

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


def backward_algorithm_recursive(
    A: List[List],
    B: List[List],
    pi: List,
    O: List,
    scaling_vector: List,
    scaled_beta_matrix: List[List] = [],
    first_iteration: bool = True,
) -> List[List]:
    if len(O) > 1:
        if first_iteration:
            bt_minus_1 = [scaling_vector[-1] for _ in A]
            scaling_vector.pop(-1)
            scaled_beta_matrix.append(bt_minus_1)

            return backward_algorithm_recursive(
                A, B, pi, O, scaling_vector, scaled_beta_matrix, False
            )
        else:
            b_temp = B[O.pop(-1)]
            beta = matrix_mulitplication(
                A, T([elem_wise_product(b_temp, scaled_beta_matrix[0])])
            )
            ct = scaling_vector.pop(-1)

            beta = [a[0] * ct for a in beta]
            scaled_beta_matrix.insert(0, beta)
            return backward_algorithm_recursive(
                A, B, pi, O, scaling_vector, scaled_beta_matrix, False
            )

    return scaled_beta_matrix

def forward_algorithm_iterative(
    A: List[List],
    B: List[List],
    pi: List,
    O: List[List],
):
    scaled_alpha_matrix = []
    scaling_vector = []

    b_t = T(B)

    alpha = elem_wise_product(pi[0], b_t[O[0]])
    c0 = 1 / sum(alpha)
    alpha = [a * c0 for a in alpha]
    scaled_alpha_matrix.append(alpha)

    for emission in O[1:]:
        alpha = elem_wise_product(
            matrix_mulitplication([scaled_alpha_matrix[-1]], A)[0], b_t[emission]
        )
        ct = 1 / sum(alpha)
        print(ct)
        alpha = [a * ct for a in alpha]
        scaling_vector.append(ct)
        scaled_alpha_matrix.append(alpha)

    return scaled_alpha_matrix, scaling_vector

def foward_algorithm_recursive(
    A: List[List],
    B: List[List],
    pi: List,
    O: List[List],
    alpha: List = [],
    scaled_alpha_matrix: List = [],
    scaling_vector: List = [],
    first_iteration: bool = True,
) -> Tuple[List[List], List[List]]:
    """Foward Algo

    Args:
        A (List[List]): Transition Matrix
        B (List[List]): Emission / Output probability matrix
        pi (list): Initial state vector
        alpha (list): vector that represents the probability of being in state j after seeing the first t observations,
        O (list): vector of emissions sequences itself
        first_iteration (bool, optional): states if the function is the first time is running. Defaults to True.

    Returns:
        float: Sum of Alpha
    """
    if len(O) > 0:
        if first_iteration:
            alpha = elem_wise_product(pi[0], T(B)[O.pop(0)])
            c0 = 1 / sum(alpha)
            alpha = [a * c0 for a in alpha]
            scaling_vector.append(c0)
            scaled_alpha_matrix.append(alpha)
            return foward_algorithm_recursive(
                A, B, pi, O, alpha, scaled_alpha_matrix, scaling_vector, False
            )
        else:
            alpha = elem_wise_product(
                matrix_mulitplication([alpha], A)[0], T(B)[O.pop(0)]
            )
            ct = 1 / sum(alpha)
            alpha = [a * ct for a in alpha]
            scaling_vector.append(ct)
            scaled_alpha_matrix.append(alpha)
            return foward_algorithm_recursive(
                A, B, pi, O, alpha, scaled_alpha_matrix, scaling_vector, False
            )
    return scaled_alpha_matrix, scaling_vector