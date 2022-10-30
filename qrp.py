# import statements:
#   sys for defining and retrieving program arguments
#   numpy to import and perform matrix operations with given data
#   scipy to perform QRP calculation
import sys as system
import numpy as numpython
from scipy import linalg


# Function: pivoted_QR, Pivoted QR factorization
# Parameters: matrix = input training data, reduce_to = number of dimensions to reduce to (default = 2)
#   To find the top 2 vectors using QR factorization to use for dimensionality reduction
def pivoted_QR(matrix, reduce_to=2):
    # Compute Q,R and P of the input matrix
    _, _, priority_vector = linalg.qr(matrix, pivoting=True)

    # Reduce the priority_vector to the length of value of 'reduce_to'
    priority_vector = priority_vector[0:reduce_to]

    # Create an empty matrix to hold the selected vectors
    rows, columns = matrix.shape
    reduced_vectors = numpython.asmatrix(numpython.empty([rows, reduce_to]))

    # Copy the prioritized columns from the input matrix to the new empty matrix we created
    for each_column in range(len(priority_vector)):
        for each_row in range(rows):
            reduced_vectors[each_row, each_column] = matrix[each_row, priority_vector[each_column]]

    # return the reduced vectors
    return reduced_vectors


# Function: reduce_data
# Parameters: v = reduced vectors, matrix = input testing data
#   To reduce the input testing_data using the top 2 selected vectors from the training_data
def reduce_data(v, matrix):
    # Return the reduced data
    return v.T * matrix


# Function: orthonormalize_vectors
# Parameters: v = vectors
#   To orthonormalize the 2 vectors to each other using modified Gram-Schmidt
def orthonormalize_vectors(v):
    # Extracting vector1
    v1 = v[:, 0]
    # Extracting vector2
    v2 = v[:, 1]

    # Ortho-normalize vector1
    ortho_normV1 = v1/numpython.linalg.norm(v1)

    # Ortho-normalize vector2
    tilda_V2 = v2 - ((ortho_normV1.T*v2).item())*ortho_normV1
    ortho_normV2 = tilda_V2/numpython.linalg.norm(tilda_V2)

    # Return the final vectors as a single np matrix
    ortho_normV = numpython.hstack((ortho_normV1, ortho_normV2))
    return ortho_normV


# Function: approximation_quality
# Parameters: v = Ortho-normalized vectors, matrix = input data
#   To compute the approximation quality of the dimensionality reduction method
def approximation_quality(v, matrix):
    # Compute the quality and return
    B_matrix = matrix * matrix.T
    v1 = v[:, 0]
    v2 = v[:, 1]
    score = v1.T*B_matrix*v1 + v2.T*B_matrix*v2
    return score


# Function: main
#   checks for arguments, imports data and calls necessary functions for dimensionality reduction
if __name__ == '__main__':
    # If the number of arguments are incorrect prompt user to rerun program and exit
    if len(system.argv) != 4:
        print('Incorrect arguments, rerun the program with correct number of arguments!')
        system.exit()

    # Exception handling for input data file
    while 1:
        try:
            training_data = numpython.asmatrix(numpython.genfromtxt(system.argv[1], delimiter=',', autostrip=True))
            testing_data = numpython.asmatrix(numpython.genfromtxt(system.argv[2], delimiter=',', autostrip=True))
            break
        except IOError:
            print('File not found')

    # Inserting an outlier into the training data
    # outlier = numpython.matrix('-3, 6, 4, -8')
    # training_data = numpython.vstack((training_data, outlier))

    # Call the required dimensionality reduction function
    vectors = pivoted_QR(training_data.T)

    # Call the function to compute the final reduced data
    reduced_data = reduce_data(vectors, testing_data.T)

    # Call the function to ortho-normalize the vectors
    on_vectors = orthonormalize_vectors(vectors)

    # Exception handling for output data file
    while 1:
        try:
            numpython.savetxt(system.argv[3], reduced_data.T, delimiter=',')
            break
        except IOError:
            print('File not written')

    # Call the function to find the quality of the algorithm
    quality = approximation_quality(on_vectors, testing_data.T)
    print(quality.item())
