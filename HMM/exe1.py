
f = open("sample_00.in", "r")
text = f.read()

def parse_input(input_value : str) -> list:
    
    """Parse input string of HMM0 from kattis

    Args:
        input_value (str): input string

    Returns:
        list: A list with the matrixes [A, B, pi]
    """

    matrixes = ()
    # print(input_value.splitlines())
    for rows in input_value.splitlines():
        values = rows.split(" ")
        dimensions = int(values[1])
        matrix_values = [float(item) for item in values[2:]]
        
        matrixes.append([matrix_values [i : i+dimensions] for i in range(0, len(matrix_values), dimensions)])
    return matrixes
