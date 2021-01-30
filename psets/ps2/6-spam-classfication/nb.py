import numpy as np

def readMatrix(file, suffix=''):
    fd = open(file+suffix, 'r')
    hdr = fd.readline()
    rows, cols = [int(s) for s in fd.readline().strip().split()]
    tokens = fd.readline().strip().split()
    matrix = np.zeros((rows, cols))
    Y = []
    for i, line in enumerate(fd):
        nums = [int(x) for x in line.strip().split()]
        Y.append(nums[0])
        kv = np.array(nums[1:])
        k = np.cumsum(kv[:-1:2])
        v = kv[1::2]
        matrix[i, k] = v
    return matrix, tokens, np.array(Y)

def nb_train(matrix, category):
    state = {}
    N = matrix.shape[1]
    ###################
    # phi1 = p{k|y=1}
    numerator = (np.sum(matrix[np.where(category==1)], axis=0)) + 1
    denominator = np.sum(matrix[np.where(category==1)]) + N
    state['phi_yeq1'] = numerator / denominator

    # phi0 = p{k|y=0}
    numerator = (np.sum(matrix[np.where(category==0)], axis=0)) + 1
    denominator = np.sum(matrix[np.where(category==0)]) + N
    state['phi_yeq0'] = numerator / denominator

    # phi = p{k|y=0}
    state['phi'] = np.sum(category==1) / N
    ###################
    return state

def nb_test(matrix, state):
    output = np.zeros(matrix.shape[0])
    ###################
    # log_phi0 = p(x|y=0), log_phi1 = p(x|y=1)
    log_phi0 = np.sum(np.log(state['phi_yeq0']) * matrix, axis=1)
    log_phi1 = np.sum(np.log(state['phi_yeq1']) * matrix, axis=1)
    phi = state['phi']
    ratio = np.exp(log_phi0 + np.log(1-phi) - log_phi1 - np.log(phi))
    prob = 1 / (1 + ratio)
    output[prob>0.5] = 1
    ###################
    return output

def evaluate(output, label):
    error = (output != label).sum() * 1. / len(output)
    print('Error: %1.4f' % error)
    return error

def main():
    trainMatrix, tokenlist, trainCategory = readMatrix('./spam_data/MATRIX.TRAIN')
    testMatrix, tokenlist, testCategory = readMatrix('./spam_data/MATRIX.TEST')

    state = nb_train(trainMatrix, trainCategory)
    output = nb_test(testMatrix, state)

    evaluate(output, testCategory)
    return

if __name__ == '__main__':
    main()
