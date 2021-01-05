from collections import namedtuple
from time import time

import torch
import numpy as np
from scipy.linalg import circulant
from tqdm import tqdm

from complex import ComplexTensor


def distributed_macropixels(X):
    height = 912
    width = 1140
    if len(X.shape) == 1:
        X = X[:, np.newaxis]

    n_samples, n_features = X.shape[0], X.shape[1]
    # in order to have a 2D input array even if there is only one input vector

    pixel_size = (height*width) // n_features

    X=X.bool()

    out = np.zeros(n_samples)
    batch_size = 3000
    n_batches = n_samples//batch_size

    for i in range(n_batches):
        batch_start, batch_end = i*batch_size, (i+1)*batch_size

        if batch_end > n_samples:
            batch_end = n_samples
            batch_size = batch_end - batch_start

        sub_out = np.zeros((batch_size, height * width), dtype=bool)

        sub_X = X[batch_start:batch_end]
        sub_out[:, :pixel_size*n_features] = np.tile(sub_X, (1, pixel_size))
        out[batch_start:batch_end] = sub_out

    return (out)

class PhaseRetriever:
    """
    Recovers the transmission matrix of the OPU.
    For now, due to the nature of the algorithm, we recover its transpose. The final TM is transposed at the end to
    return the right result.

    Parameters
    ----------
    n_row: int,
        Number of rows of the TM transposed (i.e. random features)
    n_col: int,
        Number of columns of the TM transposed.
    circ_N: int,
        size of the circulant matrices that make up the calibration signal. Higher values will increase the accuracy.
    device: str,
        device for the computation. Pick either "cpu" or "cuda:x", where x is the GPU number.
    n_signals: int,
        number of calibration signals. Higher values will increase the accuracy.
    n_anchors: int,
        number of anchors. Higher values will increase the accuracy.
    solver: str,
        solver for the retrival of the TM. Pick between "least-square" and "FFT".
    batch_size: int,
        batch size. Manges the number of rows of the TM that will be retrieved at the same time.
    verbose: int,
        if 0, prints no messages. 1 is very barebone, while 2 gives a lot of info.

    Attributes
    ----------
    n_row: int,
        Number of rows of the TM transposed (i.e. random features)
    n_col: int,
        Number of columns of the TM transposed.
    circ_N: int,
        size of the circulant matrices that make up the calibration signal. Higher values will increase the accuracy.
    device: str,
        device for the computation. Pick either "cpu" or "cuda:x", where x is the GPU number.
    n_signals: int,
        number of calibration signals. Higher values will increase the accuracy.
    n_anchors: int,
        number of anchors. Higher values will increase the accuracy.
    solver: str,
        solver for the retrival of the TM. Pick between "least-square" and "FFT".
    batch_size: int,
        batch size. Manges the number of rows of the TM that will be retrieved at the same time.
    verbose: int,
        if 0, prints no messages. 1 is very barebone, while 2 gives a lot of info.

    input_idx: named tuple,
            contains the indeces needed to separate the 3 matrices that need to be random projected.
            todo: will move it in the RP function
    time_logger: dict,
        logger for benchmarking the algorithm

    """
    def __init__(self, n_row, n_col, circ_N, device="cpu", n_signals=2, n_anchors=4, solver="fft", batch_size=32,
                 verbose=1):

        self.n_row = n_row  # rows of the TM
        self.n_col = n_col  # columns of the TM
        self.circ_N = circ_N
        self.device = device
        self.n_signals = n_signals  # would be K
        self.n_anchors = n_anchors  # would be S
        self.solver = solver
        self.batch_size = batch_size
        self.verbose = verbose

        indeces = namedtuple('indeces', ['anchors', 'diff', "anchors_diff"])
        self.input_idx = indeces(anchors=slice(0, self.n_anchors + 1),
                                 diff=slice(self.n_anchors + 1, (self.n_anchors + 1) * (self.n_signals + 1)),
                                 anchors_diff=slice((self.n_anchors + 1) * (self.n_signals + 1),
                                                    (self.n_anchors + 1) * (self.n_signals + 1) + (
                                                                self.n_anchors + 1) * (self.n_anchors + 2) // 2))

        self.time_logger = {"calibration":0, "anchor":0,"difference":0, "anchors_diff":0, "RP":0,
                            "mds":0, "lateration":0, "solver":0, "post_processing":0}

    def build_half_X(self, nonzero_ind):
        """
        Builds an half of the calibration matrix. The code has to generate two circlant matrices, concatenate them,
        then insert rows of zeros randomly to get a matrix of size (col, 2*circ_N).
        To be a bit faster, The code first generates 2 circulant matrices of size (circ_N, circ_N), gets the nonzero
        indeces of the bigger matrix and then broadcasts the calibration matrix


        Parameters
        ----------
        nonzero_ind: list of int,
            indices

        Returns
        -------
        X: numpy array,
            calibrationn matrix
        x1_first_column: numpy array,
            column of the first circulant matrix
        x2_first_column: numpy array,
            column of the second circulant matrix
        """

        X_p = [0.5, 0.5]
        x1_first_column = np.random.choice([0, 1], size=self.circ_N, p=X_p)
        X1 = circulant(x1_first_column)
        x2_first_column = np.random.choice([0, 1], size=self.circ_N, p=X_p)
        X2 = circulant(x2_first_column)

        X = np.zeros([self.n_col, 2 * self.circ_N])

        X[nonzero_ind, :self.circ_N] = X1
        X[nonzero_ind, self.circ_N:] = X2

        return X, x1_first_column, x2_first_column

    def build_X(self):
        """
        Builds the TWO calibration matrices needed to recover the final TM.

        Returns
        -------
        X.T: numpy array,
            transpose of the full calibration matrix. It is the stack of two calibration matrices. Size = ()
        first_ind: numpy array,
            FILL THIS (I think it contains the indeces where you have something for the first matrix, and 0 for the second)
        second_ind: numpy array,
            FILL THIS (I think it contains the indeces where you have something for the second matrix, and 0 for the first)
        ind_common: numpy array,
            FILL THIS (I think it contains the indeces where you have something in both matrices)

        x1, x2, x3, x4: numpy array,
            columns of the circulant matrices making up the calibration matrix. First two are for the first, the other
            two for the second.
            todo: I am pretty sure that with all the indeces that are passed, I can eliminate these and recover them later
        """

        ind = np.random.choice(self.n_col, size=2 * (self.n_col - self.circ_N), replace=False)
        first_ind = ind[:int(0.5 * len(ind))]
        second_ind = ind[int(0.5 * len(ind)):]

        ind_common = np.arange(self.n_col)[~np.isin(np.arange(self.n_col), ind)]
        first_nonzero = np.arange(self.n_col)[~np.isin(np.arange(self.n_col), first_ind)]
        second_nonzero = np.arange(self.n_col)[~np.isin(np.arange(self.n_col), second_ind)]

        X1, col1, col2 = self.build_half_X(first_nonzero)
        X2, col3, col4 = self.build_half_X(second_nonzero)

        X = np.zeros([self.n_col, 4 * self.circ_N])
        X[:, :2 * self.circ_N] = X1
        X[:, 2 * self.circ_N:] = X2

        first_ind = np.arange(self.n_col)[np.isin(np.arange(self.n_col), first_ind)]
        second_ind = np.arange(self.n_col)[np.isin(np.arange(self.n_col), second_ind)]

        return X.T, first_ind, second_ind, ind_common, col1, col2, col3, col4

    def get_eigenvalues(self, column):
        """
        Computes the eigenvalues of a circulant matrix through an FFT on a single column.
        the code returns 0.5 * the reciprocal of the eiganvalues conjugated, to avoid making this computation
        for every row later

        Parameters
        ----------
        column: torch Tensor,
            column of a circulant matrix

        Returns
        -------
        eigenvalues_reciprocal: ComplexTensor,
            eigenvalues of the circulant matrix.
        """
        fft_buffer = torch.rfft(column, signal_ndim=1, onesided=False)
        eigenvalues = ComplexTensor(real=fft_buffer[:, 0], imag=fft_buffer[:, 1])

        eigenvalues_reciprocal = 0.5 * eigenvalues.reciprocal().conj()

        return eigenvalues_reciprocal


    def make_anchors(self, calibration_matrix):
        """
        Creates the anchors signals. There is a lot of magic done here...

        First anchor has elements drawn from a bernoulli(p=0.85). Then we sum a quantity from the calibration matrix
        All the other anchors are a sum of a bernoulli(p=0.85) with the previous anchor.
        Then values above 1 are thresholded to 1
        Then we reverse the anchors. I don't get why, but you have to.

        The last anchor should always be 0, but it is not defined here. This means that the true number of anchors is
        not self.n_anchors, but (self.n_anchors + 1). This is how it is in the original code.
        todo: eliminate this! The zero anchor should be included in the anchor count.

        Parameters
        ----------
        calibration_matrix:
            calibration matrix (not the full one, the 2 partial ones).

        Returns
        -------
        anchors: numpy array,
            anchor matrix
        """
        ### Create anchor signals
        #print("WARNING: the original code takes n anchors, but then adds another 0 anchor at the end. \n TRUE NUMBER IS n_anchors +1")
        X_sum = np.sum(calibration_matrix.copy(), axis=0)
        X_sum[X_sum > 0] = 1

        anchors = np.zeros([self.n_anchors, self.n_col])

        anchor_p = [0.85, 0.15]

        anchors[0] = np.random.choice([0, 1], size=self.n_col, p=anchor_p) + X_sum
        for i in range(1, self.n_anchors):
            anchors[i] = np.random.choice([0, 1], size=self.n_col, p=anchor_p) + anchors[i - 1]

        anchors[anchors > 0] = 1

        anchors = anchors[::-1]

        return anchors

    def get_anchors_diff(self, x, anchors):
        """
        Computes the pairwise difference between anchors, WITH A TWIST!

        Not only you need the pairwise difference between anchors, but also the difference between the anchors and 1 row
        of the calibration matrix. The first (self.n_anchors + 1) elements take that into account

        Since the 0 anchor is NOT included in the original count, it is added here.
        todo: get rid of this, the 0 anchor should be there from the very beginning

        Parameters
        ----------
        x: numpy array,
            first row of the calibration matrix
        anchors: numpy array,
            anchor matrix

        Returns
        -------
        differences: numpy array,
            matrix of difference between anchors and the first row of the calibration matrix,
            plus the pairwise differences between anchors.
        """

        interfered = anchors - x
        interfered = np.vstack((interfered, x))  # x with zero (zero is less than x so subtract the other way)

        anchors = np.vstack((anchors, np.zeros(self.n_col)))  # zero is an anchor too
        differences = np.zeros(((self.n_anchors + 2) * (self.n_anchors + 1) // 2, self.n_col))

        differences[:self.n_anchors + 1] = interfered
        idx = self.n_anchors + 1

        for i in range(self.n_anchors):
            diffs = anchors[i] - anchors[1 + i:]
            differences[idx: idx + diffs.shape[0]] = diffs
            idx += diffs.shape[0]

        return differences

    def get_anchors_calibration_diff(self, X, anchors):
        """
        Computes the pairwise differences between the anchors and the calibration signals

        Since the 0 anchor is NOT included in the original count, it is added here.
        todo: get rid of this, the 0 anchor should be there from the very beginning

        Parameters
        ----------
        X: numpy array,
            calibration matrix
        anchors: numpy array
            anchor matrix

        Returns
        -------
        differences: numpy array,
            difference between all pairs of calibration and anchor signals.
        """

        differences = np.zeros((self.n_signals * (self.n_anchors + 1), self.n_col))
        diffs = 0
        for i in tqdm(range(self.n_signals)):
            idx_start, idx_end = i * (self.n_anchors + 1), (i + 1) * (self.n_anchors + 1)

            interfered = anchors - X[i]
            interfered = np.vstack((interfered, X[i]))  # x with zero (zero is less than x so subtract the other way)
            differences[idx_start:idx_end] = interfered
            diffs += interfered.shape[0]

        return differences


    def get_input_matrices(self, full_calibration_matrix, anchors_ind):
        """

        Parameters
        ----------
        full_calibration_matrix
        anchors_ind

        Returns
        -------
        torch.FloatTensor(calibration_matrix): torch Tensor,
            slice of the calibration matrix that will be used in the retrieval,

        opu_input: torch Tensor,
            stack of the 1) anchor matrix, 2) the pairwise difference between anchors and 3)the pairwise difference
             between anchors and calibration signals. These 3 will all be projected later, so they are stacked here.
        """
        start = time()
        calibration_matrix = full_calibration_matrix[anchors_ind]
        self.time_logger["calibration"] += time() - start

        start = time()
        anchors = self.make_anchors(calibration_matrix).copy()
        self.time_logger["anchor"] += time() - start

        start = time()
        differences = self.get_anchors_calibration_diff(calibration_matrix, anchors)
        self.time_logger["difference"] += time() - start

        start = time()
        anchors_diff = self.get_anchors_diff(calibration_matrix[0], anchors)
        anchors = np.vstack((anchors, np.zeros(self.n_col)))
        self.time_logger["anchors_diff"] += time() - start
        opu_input = torch.cat((torch.FloatTensor(anchors.T), torch.FloatTensor(differences.T), torch.FloatTensor(anchors_diff.T)), dim=1)


        return torch.FloatTensor(calibration_matrix), opu_input

    def project(self, A, opu_input):
        """
        Computes the random projection WITH the absolute value squared, either with the OPU or a custom matrix.
        Then it splits the stacked input into 3 matrices to better manipulate them later.

        Parameters
        ----------
        A: ComplexTensor or OPUMap object,
            either a random matrix or the OPUMap object.
        opu_input: torch tensor,
            matrix to random project.

        Returns
        -------
        anchors_RP: torch.Tensor,
            Random projection of the anchors.
        difference_RP: torch.Tensor,
            Random projection of the difference between anchors and calibration matrices.
        anchors_diff_RP: torch.Tensor,
            Random projection of the pairwise difference between anchors.
        """
        start = time()

        if type(A) == ComplexTensor:
            prod = (A @ opu_input).abs() ** 2
        else:
            prod = A(opu_input.bool().T).T.float()

        """
        if self.verbose == 1:
            print((prod > 255).sum())
            saturation = (prod > 255).sum().float()/prod.numel() * 100
            print("Saturation at {0:3.3f}%".format(saturation))
        """

        anchors_RP = prod[:, self.input_idx.anchors]
        difference_RP = prod[:, self.input_idx.diff]
        anchors_diff_RP = prod[:, self.input_idx.anchors_diff]

        proj_time = time() - start

        self.time_logger["RP"] += proj_time

        return anchors_RP, difference_RP, anchors_diff_RP

    def mds(self, anchors_diff_RP):
        """
        Creates a difference matrix for the anchors, then performs classical multidimensional scaling to recover
        the anchors' real and imaginary part.

        Parameters
        ----------
        anchors_diff_RP: torch tensor,
            Random projection with modulus square of the pairwise difference of the anchors.
             Shape = (n_rows * anchors*(anchors - 1)/2).

        Returns
        -------
        anchor: ComplexTensor,
            reconstructed anchors.

        NOTE: In the diagonalization, 2 eigenvalues should be dominant, while the others should be close to zero.
        For some reason, numpy catches this much better than torch: the "0" of numpy is 1e*-14, while for torch is 1e-6

        NOTE 2: in pytorch, symeig returns the eigenvalues in crescent order. That is why the indeces are reversed when
        assigning the real and imaginary part at the end.

        NOTE 3: maybe move the computation of the difference matrix outside the class.

        THE BIGGEST NOTE: The distance matrix at this point has size (self.n_anchors + 2, self.n_anchors + 2), even
        though we have self.n_anchors. That is because we added the distances between all anchors and the origin at the
        end, and the distance between all anchors and a row of the calibration matrix at the beginning.
        When grabbing the recovered anchors, the first will be discarded, because that will be the row of the calibration
        matrix in the complex space, which we do not need.
        """
        start = time()

        upper_idx = torch.triu_indices(self.n_anchors + 2, self.n_anchors + 2, offset=1)

        distances = torch.zeros(anchors_diff_RP.shape[0], self.n_anchors + 2, self.n_anchors + 2)
        for row in range(anchors_diff_RP.shape[0]):
            distances[row, upper_idx[0, :], upper_idx[1, :]] = anchors_diff_RP[row, :]
            distances[row] = distances[row] + distances[row].T
        # The centering matrix can be generated just once
        centering_matrix = torch.eye(self.n_anchors+2) - 1. / (self.n_anchors + 2) * torch.ones((self.n_anchors + 2, self.n_anchors + 2))

        normalized_distances = -0.5 * centering_matrix @ distances @ centering_matrix

        diagonalization = torch.symeig(normalized_distances, eigenvectors=True)

        eigval = diagonalization.eigenvalues[:,-2:]
        eigvec = diagonalization.eigenvectors[:,:, -2:]

        eigval = eigval.reshape(-1)

        eigvec = eigvec.permute(1, 0, 2).reshape(self.n_anchors + 2, -1)

        rec_anchors = torch.sqrt(eigval) * eigvec
        rec_anchors = rec_anchors - rec_anchors[-1]

        # distances.shape[0] is the batch size. Since the last batch can have less elements, I use that instead.
        rec_anchors = rec_anchors.reshape(self.n_anchors + 2, distances.shape[0], 2).permute((1, 0, 2))

        anchor = ComplexTensor(real=rec_anchors[:, 1:, 1], imag=rec_anchors[:, 1:, 0])

        self.time_logger["mds"] += time() - start
        return anchor


    def get_calibration_phase(self,anchors_RP, difference_RP, anchors_diff_RP):
        """
        Computes the real and imaginary part of the signal in the complex space (Y in the paper).

        Parameters
        ----------
        anchors_RP: torch.Tensor,
            Random projection of the anchors.
        difference_RP: torch.Tensor,
            Random projection of the difference between anchors and calibration matrices.
        anchors_diff_RP: torch.Tensor,
            Random projection of the pairwise difference between anchors.

        Returns
        -------
        signal: ComplexTensor,
            Tensor with the signal values

        """

        LHS = difference_RP - anchors_RP.unsqueeze(-1)

        anchor_no_square = self.mds(anchors_diff_RP)

        # The matrix of ones can be cached to avoid generation at every step. However, do be midful
        # the number of rows is not divisible by the batch size

        RHS = torch.stack((-2 * anchor_no_square.real, -2 * anchor_no_square.imag,
                           torch.ones(anchor_no_square.shape()[0], self.n_anchors + 1)), dim=2)

        start = time()

        lateration_solution = torch.pinverse(RHS) @ LHS
        signal = ComplexTensor(real=lateration_solution[:, 0, :], imag=lateration_solution[:, 1, :])
        self.time_logger["lateration"] += time() - start
        return signal


    def reconstruct_matrix(self, signal, calibration_matrix, eigval1_reciprocal, eigval2_reciprocal):
        """
        Recovers the random matrix.

        Parameters
        ----------

        signal: ComplexTensor,
            Tensor with the signal values
        calibration_matrix: torch.Tensor,
            calibration matrix (the partial one)
        eigenvalues_reciprocal1: ComplexTensor,
            eigenvalues of the first circulant matrix block of the partial calibration matrix.
        eigenvalues_reciprocal2: ComplexTensor,
            eigenvalues of the second circulant matrix block of the partial calibration matrix.

        Returns
        -------
        reconstructed_A: ComplexTensor,
            recostructed transmission matrix. If batch size<rows, it is a batch of rows.

        """
        start = time()

        if self.solver == "least-square":

            inv_calibration_matrix = torch.pinverse(calibration_matrix)
            reconstructed_A = ComplexTensor(real=torch.matmul(signal.real, inv_calibration_matrix),
                                            imag=torch.matmul(signal.imag, inv_calibration_matrix))


        elif self.solver == "fft":
            signal = signal.conj().stack()

            signal1_star = signal[:, :self.n_signals//2]
            signal2_star = signal[:, self.n_signals//2:]


            fft_buffer = torch.fft(signal1_star, signal_ndim=1)

            block1 = ComplexTensor(real=fft_buffer[:,:, 0], imag=fft_buffer[:,:, 1])
            block1 = block1.batch_elementwise(eigval1_reciprocal)


            fft_buffer = torch.fft(signal2_star, signal_ndim=1)

            block2 = ComplexTensor(real=fft_buffer[:, :, 0], imag=fft_buffer[:, :, 1])

            block2 = block2.batch_elementwise(eigval2_reciprocal)

            reconstructed_A = torch.ifft((block1 + block2).stack(), signal_ndim=1)
            reconstructed_A = ComplexTensor(real=reconstructed_A[:,:, 0], imag=reconstructed_A[:, :, 1]).conj()

        self.time_logger["solver"] += time() - start

        return reconstructed_A

    def recover_matrix(self, A, full_calibration_matrix, anchors_ind, col1, col2):
        """

        partially recovers the transmission matrix (TM) of the OPU. This requires launching the algorithm twice,
         with two sets of calibration and anchor matrices.

        Parameters
        ----------
        A: ComplexTensor or OPUMap object,
            either a random matrix or the OPUMap object.
        full_calibration_matrix: numpy array,
            full calibration matrix. The necessary slice will be obtained here.
        anchors_ind: numpy array,
            contains the indeces of the slice of the TM to use in the recovery.
        col1: numpy array,
            column of the first circulant matrix
        col2: numpy array,
            column of the second circulant matrix

        Returns
        -------
        reconstructed_A: ComplexTensor,
            partially reconstructed transmission matrix.
        """
        reconstructed_A = ComplexTensor(real=torch.zeros(self.n_row, self.circ_N),
                                        imag=torch.zeros(self.n_row, self.circ_N))

        calibration_matrix, opu_input = self.get_input_matrices(full_calibration_matrix, anchors_ind)
        eigval1_reciprocal = self.get_eigenvalues(torch.FloatTensor(col1))
        eigval2_reciprocal = self.get_eigenvalues(torch.FloatTensor(col2))

        anchors_RP, difference_RP, anchors_diff_RP = self.project(A, opu_input)
        difference_RP = difference_RP.reshape(self.n_row, -1, self.n_anchors + 1).transpose(1, 2)


        if self.verbose:
            print("Recovering rows...")

        n_batches = torch.ceil(torch.tensor(self.n_row / self.batch_size)).type(torch.int).item()

        for batch_idx in tqdm(range(n_batches)):
            batch_start, batch_end = self.batch_size * batch_idx, self.batch_size * (batch_idx + 1)

            signal = self.get_calibration_phase(anchors_RP[batch_start:batch_end],
                                                difference_RP[batch_start:batch_end],
                                                anchors_diff_RP[batch_start:batch_end])

            rec_batch = self.reconstruct_matrix(signal, calibration_matrix, eigval1_reciprocal, eigval2_reciprocal)

            reconstructed_A[batch_start:batch_end, :] = rec_batch

        return reconstructed_A


    def fit(self, A):
        """
        Partially recovers the transmission matrix, then combines the result to get the fully recovered matrix.
        todo: the combined part can be rewritten in torch to speed things up a bit.

        Parameters
        ----------
        A: ComplexTensor or OPUMap object,
            either a random matrix or the OPUMap object.

        Returns
        -------
        rec_A: ComplexTensor,
            Fully reconstructed transmission matrix
        """
        full_calibration_matrix, first_ind, second_ind, ind_common, col1, col2, col3, col4 = self.build_X()

        anchors1_ind = np.arange(self.n_signals)
        anchors2_ind = np.arange(self.n_signals, 2 * self.n_signals)
        with torch.no_grad():
            rec_A1 = self.recover_matrix(A, full_calibration_matrix, anchors1_ind, col1, col2)

            rec_A2 = self.recover_matrix(A, full_calibration_matrix, anchors2_ind, col3, col4)

            start = time()
            A1_ind = np.arange(self.n_col)[~np.isin(np.arange(self.n_col), first_ind)]
            A2_ind = np.arange(self.n_col)[~np.isin(np.arange(self.n_col), second_ind)]


            A1 = ComplexTensor(real=torch.zeros(self.n_row, self.n_col), imag=torch.zeros(self.n_row, self.n_col))
            A2 = ComplexTensor(real=torch.zeros(self.n_row, self.n_col), imag=torch.zeros(self.n_row, self.n_col))

            A1[:, A1_ind] = rec_A1
            A2[:, A2_ind] = rec_A2

            A1 = A1.numpy()
            A2 = A2.numpy()

            P1 = np.angle(A1[:, ind_common] / A2[:, ind_common]).astype('float32')
            P2 = np.angle(A1[:, ind_common] / np.conj(A2[:, ind_common])).astype('float32')

            mean_P1 = np.mean(P1, axis=1)
            mean_P2 = np.mean(P2, axis=1)

            P1 = np.std(P1, axis=1)
            P2 = np.std(P2, axis=1)

            mask1 = P1 < P2
            mask2 = np.invert(mask1)
            A2[mask2] = np.conj(A2[mask2])
            phases = (mean_P1 * mask1 + mean_P2 * mask2).reshape([self.n_row, 1])
            phases = np.exp(1j * phases)
            A2 = phases * A2

            A1_ind = first_ind  # indices which are all zero
            A1[:, A1_ind] = A2[:, A1_ind]

            rec_A = ComplexTensor(real=torch.FloatTensor(np.real(A1)), imag=torch.FloatTensor(np.imag(A1)))
            self.time_logger["post_processing"] = time() - start
        return rec_A