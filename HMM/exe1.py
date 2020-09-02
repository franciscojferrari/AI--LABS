
import sys



def parse_input(input_value : str) -> list:
    
    """Parse input string of HMM0 from kattis

    Args:
        input_value (str): input string

    Returns:
        list: A list with the matrixes [A, B, pi]
    """

    matrixes = []
    # print(input_value.splitlines())
    for rows in input_value.splitlines():
        values = rows.split(" ")
        dimensions = int(values[1])
        matrix_values = [float(item) for item in values[2:]]
        
        matrixes.append([matrix_values [i : i+dimensions] for i in range(0, len(matrix_values), dimensions)])
    return matrixes


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


def parse_output(matrix:list)->str:
    rows = len(matrix)
    columns = len(matrix[0])
    list = [rows, columns] + [item for row in matrix for item in row]

    print (' '.join(map(str, list)))


def main():
    # print command line arguments
    file_content = ""
    for text in sys.stdin:
       file_content+=text
    A, B, pi = parse_input(file_content)
    parse_output(matrix_mulitplication(matrix_mulitplication(pi, A), B))

if __name__ == "__main__":
    main()