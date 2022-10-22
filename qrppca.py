# import statements:
#   sys for defining and retrieving program arguments
#   numpy to import and perform matrix operations with given data
#   scipy to perform QRP calculation
import sys as system
import numpy as numpython
from scipy import linalg


# Function: centered_PCA, Principal Component Analysis with mean subtraction
# Parameters: matrix = input training data, reduce_to = number of dimensions to reduce to (default = 2)
#   To find the top 2 eigen vectors using PCA with mean subtraction to use for dimensionality reduction
def centered_PCA(matrix, reduce_to=2):
    # Compute average and subtract it from the input matrix
    avg = numpython.mean(matrix, axis=1)
    matrix_avg = matrix - avg

    # Compute the symmetric matrix B for PCA
    B_matrix = matrix_avg * matrix_avg.T

    # Compute eigen values and vectors of the symmetric matrix B
    eigen_values, eigen_vectors = numpython.linalg.eigh(B_matrix)

    # sort the eigen values and vectors in decreasing order of the eigen values
    pivot = numpython.argsort(eigen_values)[::-1]
    eigen_values = eigen_values[pivot]
    eigen_vectors = eigen_vectors[:, pivot]

    # Find the top 2 eigen vectors
    reduced_eigen_vectors = eigen_vectors[:, :reduce_to]

    # return the reduced vectors
    return reduced_eigen_vectors


# Function: pivoted_QR, Pivoted QR factorization
# Parameters: matrix = input training data, reduce_to = number of dimensions to reduce to (default = 2)
#   To find the top k vectors using QR factorization to use for dimensionality reduction
def pivoted_QR(matrix, k):
    # Compute Q,R and P of the input matrix
    _, _, priority_vector = linalg.qr(matrix, pivoting=True)

    # Reduce the priority_vector to the length of value of 'reduce_to'
    priority_vector = priority_vector[0:k]

    # Create an empty matrix to hold the selected vectors
    rows, columns = matrix.shape
    reduced_vectors = numpython.asmatrix(numpython.empty([rows, k]))

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
def approximation_quality(v, k_matrix, matrix):
    # Compute average and subtract it from the input matrix
    avg_tilda = numpython.mean(k_matrix, axis=1)
    avg = numpython.mean(matrix, axis=1)
    matrix_avg = matrix - avg_tilda
    _, col = matrix.shape

    # Compute the quality and return
    B_matrix = matrix_avg * matrix_avg.T
    # B_matrix = B_matrix/(numpython.linalg.norm(B_matrix)**2)
    v1 = v[:, 0]
    v2 = v[:, 1]
    score = v1.T*B_matrix*v1 + v2.T*B_matrix*v2 + 2*col*avg.T*avg_tilda - col*(numpython.linalg.norm(avg_tilda)**2)
    return score


# Function: main
#   checks for arguments, imports data and calls necessary functions for dimensionality reduction
if __name__ == '__main__':
    # If the number of arguments are incorrect prompt user to rerun program and exit
    if len(system.argv) != 5:
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
    # outlier = numpython.matrix('-36356356356, 6363634, 46436, -8984508240582082')
    # training_data = numpython.vstack((training_data, outlier))

    # If the value of k is more than training data rows prompt user to rerun program with correct value and exit
    input_rows, input_columns = training_data.shape
    if int(system.argv[4]) > input_rows:
        print('Incorrect value for k, rerun with correct value, should be less than', input_rows)
        system.exit()

    # Call pivoted_Qr function to retrieve top k columns for dimensionality reduction
    k_data = pivoted_QR(training_data.T, int(system.argv[4]))

    # Call the required dimensionality reduction function
    vectors = centered_PCA(k_data)

    # Call the function to ortho-normalize the vectors
    on_vectors = orthonormalize_vectors(vectors)

    # Call the function to compute the final reduced data
    reduced_data = reduce_data(on_vectors, testing_data.T)

    # Exception handling for input data file
    while 1:
        try:
            numpython.savetxt(system.argv[3], reduced_data.T, delimiter=',')
            break
        except IOError:
            print('File not written')

    # Call the function to find the quality of the algorithm
    quality = approximation_quality(on_vectors, k_data, testing_data.T)
    print(quality.item())
