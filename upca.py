# import statements:
#   sys for defining and retrieving program arguments
#   numpy to import and perform matrix operations with given data
import sys as system
import numpy as numpython


# Function: uncentered_PCA, Principal Component Analysis without mean subtraction
# Parameters: matrix = input training data, reduce_to = number of dimensions to reduce to (default = 2)
#   To find the top 2 eigen vectors and reduce the dimensions of given data to 2 using PCA without mean subtraction
def uncentered_PCA(matrix, reduce_to=2):
    # Compute the symmetric matrix B for PCA
    B_matrix = matrix * matrix.T

    # Compute eigen values and vectors of the symmetric matrix B
    eigen_values, eigen_vectors = numpython.linalg.eigh(B_matrix)

    # sort the eigen values and vectors in decreasing order of the eigen values
    pivot = numpython.argsort(eigen_values)[::-1]
    eigen_values = eigen_values[pivot]
    eigen_vectors = eigen_vectors[:, pivot]

    # Find the top 2 eigen vectors
    reduced_eigen_vectors = eigen_vectors[:, :reduce_to]

    # return the reduced vectors v1 and v2 of size (1, 150) each and the reduced_matrix with just 2 rows
    return reduced_eigen_vectors


# Function: reduce_data
# Parameters: v = reduced eigen vectors, matrix = input testing data
#   To reduce the input testing_data using the top 2 selected eigen vectors from the training_data
def reduce_data(v, matrix):
    # Return the reduced data
    return v.T * matrix


# Function: orthonormalize_vectors
# Parameters: v = eigen vectors
#   To orthonormalize the 2 eigen vectors to each other using modified Gram-Schmidt
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
# Parameters: v = Ortho-normalized eigen vectors, matrix = input data
#   To orthonormalize the 2 eigen vectors to each other using modified Gram-Schmidt
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
            training_data = numpython.asmatrix(numpython.genfromtxt(system.argv[1], delimiter=','))
            testing_data = numpython.asmatrix(numpython.genfromtxt(system.argv[2], delimiter=','))
            break
        except IOError:
            print('File not found')

    # Inserting an outlier into the training data
    # outlier = numpython.matrix('-36356356356, 6363634, 46436, -8984508240582082')
    # training_data = numpython.vstack((training_data, outlier))

    # Call the required dimensionality reduction function
    vectors = uncentered_PCA(training_data.T)

    # Call the function to compute the final reduced data
    reduced_data = reduce_data(vectors, testing_data.T)

    # Exception handling for input data file
    while 1:
        try:
            numpython.savetxt(system.argv[3], reduced_data.T, delimiter=',')
            break
        except IOError:
            print('File not written')

    # Call the function to ortho-normalize the eigen vectors
    on_vectors = orthonormalize_vectors(vectors)

    # Call the function to find the quality of the algorithm
    quality = approximation_quality(on_vectors, testing_data.T)
    print(quality.item())
