import sys
import itertools
RECURSIVE = True


def elem_wise_product(vector_a:list, vector_b:list)->list:
    """Element-wise product between 2 vectors

    Args:
        vector_a (list): Vector A
        vector_b (list): Vector C

    Returns:
        list: New vector with the result of the element-wise multiplication between A and B
    """
    return [(element_a * element_b) for element_a , element_b in zip(vector_a, vector_b)]

def elem_wise_product_matrix(matrix_a:list, matrix_b:list)->list:
    """Element-wise product between 2 matrices

    Args:
        matrix_a (list): Matrix A (dimension n, m)
        matrix_b (list): Matrix B (dimension n, m)

    Returns:
        list:  Element-wise multiplication between A and B (dimension n, m)
    """
    n_columns = len(matrix_a[0])
    result = [a * b for row_a, row_b in zip(matrix_a, matrix_b) for a, b in zip(row_a, row_b)]

    return [result[i : i + n_columns] for i in range(0, len(result), n_columns)] 

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
        list: A list with the matrixes [A, B, pi, O]
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
    """Transpose of matrix

    Args:
        matrix (list): Matrix dimension n, m

    Returns:
        list: Transponsed matrix of dimension m, n
    """
    return list(map(list, zip(*matrix)))

def foward_algorithm_recursive (A:List[List], B:List[List], pi:list, alpha:list, O:list, first_interation:bool = True) -> float:
    """Foward Algo

    Args:
        A (List[List]): Transition Matrix
        B (List[List]): Emission / Output probability matrix
        pi (list): Initial state vector
        alpha (list): vector that represents the probability of being in state j after seeing the first t observations,
        O (list): vector of emissions sequences itself
        first_interation (bool, optional): states if the function is the first time is running. Defaults to True.

    Returns:
        float: Sum of Alpha
    """
    if len(O) > 0:
        if first_interation:
            alpha = elem_wise_product(pi[0], T(B)[O.pop(0)])
            return foward_algorithm_recursive(A, B, pi, alpha, O, False)
        else:
            alpha = elem_wise_product( matrix_mulitplication([alpha] , A)[0], T(B)[O.pop(0)])
            return foward_algorithm_recursive(A, B, pi, alpha, O, False)
    print(sum(alpha))

def foward_algorithm (A:List[List], B:List[List], pi:list, O:list) -> float:
    """[summary]

    Args:
        A (List[List]): Transition Matrix
        B (List[List]): Emission / Output probability matrix
        pi (list): Initial state vector
        alpha (list): vector that represents the probability of being in state j after seeing the first t observations,
        O (list): vector of emissions sequences itself

    Returns:
        float: [description]
    """
    for index, emission in enumerate(O):
        if index == 0:
            alpha = elem_wise_product(pi[0], T(B)[emission])
        else:
            alpha =elem_wise_product( matrix_mulitplication([alpha], A)[0], T(B)[emission])
    print(sum(alpha))

def viterbi_algorithm(A:list, B:list, pi:list, O:list) -> list :
    """[summary]

    Args:
        A (List[List]): Transition Matrix
        B (List[List]): Emission / Output probability matrix
        pi (list): Initial state vector
        alpha (list): vector that represents the probability of being in state j after seeing the first t observations,
        O (list): vector of emissions sequences itself


    Returns:
        list: [description]
        list: [description]
    """

    theta_list = []
    theta_idx_list = []
    B_T = T(B)
    A_T = T(A)

    for index, observation_t in enumerate(O):
        if index == 0:
            theta_i = elem_wise_product(pi[0], T(B)[observation_t])
        else:
            theta_m = [theta_i for _ in theta_i] #Copy vector theta_i to create a new matrix of size n, n
            obs_t_m_T = T([B_T[observation_t] for _ in theta_i]) #Get the overservation (vector) --> copy it to create matrix of size theta --> transpose it to get same values per row
                #[0.6, 0.1, 0.0, 0.0] ->[[0.6, 0.1, 0.0, 0.0], [0.6, 0.1, 0.0, 0.0], [0.6, 0.1, 0.0, 0.0], [0.6, 0.1, 0.0, 0.0]] -->[[0.6, 0.6, 0.6, 0.6], [0.1, 0.1, 0.1, 0.1], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
            result =  elem_wise_product_matrix( elem_wise_product_matrix(A_T, theta_m), obs_t_m_T )
            #find the max
            theta_i = ([max(row) for row in result]) #find theta by comuting the argmax between each element in the rows of the result matrix 
            theta_idx = [row.index( max(row)) for row in result] #find theta idx by comuting the argmax between each element in the rows of the result matrix and geting the index

            theta_list.append(theta_i)
            theta_idx_list.append(theta_idx)

    return theta_list, theta_idx_list

def viterbi_algorithm_optimal_sequence(theta_list:list, theta_idx_list:list, sequence:list = None)->list:
    """[summary]

    Args:
        theta_list (list): [description]
        theta_idx_list (list): [description]
        sequence (list, optional): [description]. Defaults to None.

    Returns:
        list: [description]
    """
    ##get the max
    if len(theta_idx_list) > 0:
        if not sequence:
            index_max_value = theta_list[-1].index( max(theta_list[-1]) )
            sequence = [index_max_value]
            return viterbi_algorithm_optimal_sequence(theta_list, theta_idx_list, sequence)
        else:
            # print(sequence)
            next_idx = theta_idx_list.pop()[sequence[-1]]
            sequence.append(next_idx)
            return viterbi_algorithm_optimal_sequence(theta_list, theta_idx_list, sequence)
    else:
        return sequence
    
def print_list(input_list:list)->str:
    return_list = [str(i) for i in input_list[::-1]]
    print( " ".join(return_list))

def main():
    file_content = "".join([text for text in sys.stdin])
    A, B, pi, O = parse_input(file_content)
    theta_list, theta_idx_list = viterbi_algorithm(A, B, pi, O)
    optimal_seq = viterbi_algorithm_optimal_sequence(theta_list, theta_idx_list)
    # print(theta_list, theta_idx_list)
    print_list(optimal_seq)

if __name__ == "__main__":
    main()
