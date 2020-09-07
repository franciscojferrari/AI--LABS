import sys
import itertools
import math
from typing import List
RECURSIVE = True
sys.setrecursionlimit(10000)


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

def foward_algorithm_recursive (A:List[List], B:List[List], pi:list, O:List[List], alpha:list = [], scaled_alpha_matrix:list = [], scaling_vector:list = [], first_iteration:bool = True,) -> float:
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
            alpha = [a*c0 for a in alpha]
            scaling_vector.append(c0)
            scaled_alpha_matrix.append(alpha)
            return foward_algorithm_recursive(A, B, pi, O, alpha, scaled_alpha_matrix, scaling_vector, False)
        else:
            alpha = elem_wise_product( 
                matrix_mulitplication([alpha] , A)[0],
                 T(B)[O.pop(0)]
                )
            ct = 1 / sum(alpha)
            alpha = [a*ct for a in alpha]
            scaling_vector.append(ct)
            scaled_alpha_matrix.append(alpha)
            return foward_algorithm_recursive(A, B, pi, O, alpha, scaled_alpha_matrix, scaling_vector, False)
    # print(scaled_alpha_matrix)
    # print("********")
    # print(len(scaled_alpha_matrix))
    # print(len(scaling_vector))
    # print("********")
    return scaled_alpha_matrix, scaling_vector

def backward_algorithm_recursive (A:List[List], B:List[List], pi:list, O:list, scaling_vector:list, scaled_beta_matrix:List[List] = [], first_iteration:bool = True,) -> float:
    bt_minus_1 = [scaling_vector[-1] for _ in A]
    scaled_beta_matrix.append(bt_minus_1)

    for t, emission in reversed(list(enumerate(O[:-1]))):
        beta = []
        for i in range(len(A)):
            beta_temp = 0
            for j in range(len(A)):
                beta_temp +=  A[i][j] * B[j][O[t+1]] * scaled_beta_matrix[0][j]
            beta_temp = beta_temp * scaling_vector[t]
            beta.append(beta_temp)
        scaled_beta_matrix.insert(0, beta)
    return scaled_beta_matrix
    # if len(O)>1:
    #     if first_iteration:
    #         bt_minus_1 = [scaling_vector[-1] for _ in A]
    #         scaling_vector.pop(-1)
    #         scaled_beta_matrix.append(bt_minus_1)
            
    #         return backward_algorithm_recursive(A, B, pi, O, scaling_vector, scaled_beta_matrix, False)
    #     else:
    #         beta = elem_wise_product(
    #             matrix_mulitplication(A, [B[O.pop(-1)]])[0], scaled_beta_matrix[0]
    #             )
    #         print("------", beta)
    #         # beta = [a*scaled_beta_matrix[0] for a in matrix_mulitplication(A, [B[O.pop(-1)]])[0]]
    #         ct = scaling_vector.pop(-1)
    #         beta = [a * ct for a in beta]
    #         scaled_beta_matrix.insert(0, beta)
    #         return backward_algorithm_recursive(A, B, pi, O, scaling_vector, scaled_beta_matrix, False)
    # return scaled_beta_matrix            


def di_gamma_algorithm (A, B, O, scaled_alpha_matrix, scaled_beta_matrix, gamma_list = [], di_gamma_list = []):
    # print(len(A))
    for t in range(len(O[:-1])):
        di_gamma = []
        for i in range(len(A)):
            di_gamma.append([])
            for j in range(len(A)):
                di_gamma_temp = scaled_alpha_matrix[t][i] * A[i][j] * B[j][O[t+1]] * scaled_beta_matrix[t+1][j]
                di_gamma[-1].append(di_gamma_temp)
        gamma = [ sum(row) for row in di_gamma]
        gamma_list.append(gamma)
        di_gamma_list.append(di_gamma)

    gamma_list.append(scaled_alpha_matrix[-1])
    return gamma_list, di_gamma_list

    # if len(O) == 1:
    #     gamma_list.append(scaled_alpha_matrix[-1])
    #     return gamma_list, di_gamma_list
    # else:
    #     di_gamma = []
    #     emmission_t = O.pop(0)
    #     current_alpha = scaled_alpha_matrix.pop(0)
    #     current_betta = scaled_beta_matrix.pop(0)
    #     for i in enumerate(len(A)):
    #         di_gamma.append([])
    #         for j in enumerate(len(A)):
    #             di_gamma_temp = current_alpha[i] * A[i][j] * B[j][emmission_t] * current_betta[j]
    #             di_gamma[-1].append(di_gamma_temp)
    #     gamma = [ sum(row) for row in di_gamma]
    #     gamma_list.append(gamma)
    #     di_gamma_list.append(di_gamma)
    #     return di_gamma_algorithm(A, B, O, scaled_alpha_matrix, scaled_beta_matrix, gamma_list, di_gamma_list)

def re_estimate_pi(gamma_list):
    return [gamma_list[0]]

def re_estimate_A(A, gamma_list, di_gamma_list):
    re_estimated_A = []
    for i in range(len(A)):
        re_estimated_A.append([])
        # print(gamma_list)
        denom = sum([row[i] for row in gamma_list[::-1]])
        for j in range(len(A)):
            number = sum([matrix[i][j] for matrix in di_gamma_list])
            re_estimated_A[-1].append(number/denom)
    return re_estimated_A

def re_estimate_B(B, O, gamma_list, di_gamma_list):
    re_estimated_B = []
    # print("---------")
    # print(len(B))
    # print(len(O))
    # print(len(gamma_list))
    # print(len(di_gamma_list))
    # print("---------")
    for i in range(len(B)):
        re_estimated_B.append([])
        denom = sum([row[i] for row in gamma_list])
        for j in range(len(B[0])):
            number = sum([vector[i] for t, vector in enumerate(gamma_list) if O[t] == j])
            re_estimated_B[-1].append(number/denom)
    return re_estimated_B

def log_PO_given_lambda(scaling_vector):
    return -sum([math.log(ci) for ci in scaling_vector])

def find_model(pi, A, B, O, maxIters):
    iters = 0
    logProb = -99999999999
    oldLogProb = -math.inf

    # while (iters < maxIters and logProb > oldLogProb):
    logProb_check = 0
    saved_A, saved_B = [], []
    while (iters < maxIters):
        # if logProb > oldLogProb:
        #     saved_A, saved_B = A, B
        #     logProb_check = 0  
        # else:
        #     logProb_check += 1
        if  1 > (oldLogProb - logProb) and (oldLogProb - logProb) > 0:
            return A, B

        # print(logProb_check)
        oldLogProb = logProb
        scaled_alpha_matrix, scaling_vector = foward_algorithm_recursive(A, B, pi, O.copy(), [], [], [])

        scaled_beta_matrix = backward_algorithm_recursive(A, B, pi, O.copy(), scaling_vector.copy())
        
        gamma_list, di_gamma_list  = di_gamma_algorithm(A, B, O.copy(), scaled_alpha_matrix.copy(), scaled_beta_matrix.copy(), [], [])
        
        # print(pi)
        # print(A)
        # print(iters)
        # print(sum(pi[0]))
        pi = re_estimate_pi(gamma_list)
        A = re_estimate_A(A, gamma_list, di_gamma_list)
        B = re_estimate_B(B, O, gamma_list, di_gamma_list)
        
        # print(A)
        # print(B)
        logProb = log_PO_given_lambda(scaling_vector)
        iters += 1
        # print(oldLogProb - logProb)
        # break
        # print(iters)
    return A, B

def print_list(input_list:list)->str:
    return_list = [str(i) for i in input_list[::-1]]
    print( " ".join(return_list))

def parse_matrix(matrix:list)->str:
    rows = len(matrix)
    columns = len(matrix[0])
    list = [rows, columns] + [item for row in matrix for item in row]

    print (' '.join(map(str, list)))

def main():
    file_content = "".join([text for text in sys.stdin])
    A, B, pi, O = parse_input(file_content)
    # B = [int(element) for row in B for element in row]
    # print(find_model(pi, A, B, O, 100))
    new_A, new_B = find_model(pi, A, B, O, 50)
    parse_matrix(new_A)
    parse_matrix(new_B)
    

if __name__ == "__main__":
    main()
