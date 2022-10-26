# import statements:
#   sys for defining and retrieving program arguments
#   numpy to import and perform matrix operations with given data
#   random to perform random feature selection
import sys as system
import numpy as numpython
import random as rn


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


# Function: fkv, Randomized feature selection
# Parameters: matrix = input training data, k = number of columns to select
#   To find the top 2 vectors using FKV algorithm to use for dimensionality reduction
def fkv(matrix, k):
    # Create a vector that has the probability weights for each column of the input data
    rows, columns = matrix.shape
    weights = numpython.empty(columns)
    sum_norm_sq = 0
    for i in range(columns):
        weights[i] = numpython.linalg.norm(matrix[:, i])**2
        sum_norm_sq += weights[i]
    weights /= sum_norm_sq

    # Randomly select the required number of columns
    reduced_vectors = numpython.asmatrix(numpython.empty([rows, k]))
    selected_vectors = rn.choices(list(range(0, columns)), weights, k=k)
    for each_column in range(len(selected_vectors)):
        for each_row in range(rows):
            reduced_vectors[each_row, each_column] = matrix[each_row, selected_vectors[each_column]]

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
# Parameters: v = Ortho-normalized vectors, k_matrix = matrix with k rows, matrix = input data
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
    if len(system.argv) != 6:
        print('Incorrect arguments, rerun the program with correct number of arguments!')
        system.exit()

    # Set random seed from the input arguments
    rn.seed(system.argv[5])

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

    random_iterations = 1
    while random_iterations > 0:
        # Call fkv function to retrieve top k columns for dimensionality reduction
        k_data = fkv(training_data.T, int(system.argv[4]))

        # Call the required dimensionality reduction function
        vectors = centered_PCA(k_data)

        # Call the function to ortho-normalize the vectors
        on_vectors = orthonormalize_vectors(vectors)

        # Call the function to compute the final reduced data
        reduced_data = reduce_data(on_vectors, testing_data.T)

        # Exception handling for output data file
        while 1:
            try:
                numpython.savetxt(system.argv[3], reduced_data.T, delimiter=',')
                break
            except IOError:
                print('File not written')

        # Call the function to find the quality of the algorithm
        quality = approximation_quality(on_vectors, k_data, testing_data.T)
        print(quality.item())
        random_iterations -= 1