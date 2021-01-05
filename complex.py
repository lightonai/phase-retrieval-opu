"""
This script contains a class which adds support to the complex dtype in Pytorch. It is by no means perfect,
and it has not been tested with things like backprop. However it can work in a similar way to the numpy complex dtype,
with the added advantage that it can exploit the batching and GPU features available in Pytorch.

Only the operations needed in the phase retrieval algorithm have been implemented, like sum, products or differences.
Others can be implenented on requests, anything that is possible with numpy should be possible here.
"""

from collections import namedtuple

import torch
import numpy as np

class ComplexTensor:
    """
    Parameters/Attributes
    ---------------------
    real: torch.FloatTensor,
        real part of the array
    imag: torch.FloatTensor,
        imaginary part of the array
    """
    def __init__(self, real, imag):
        self.real = real
        self.imag = imag

    def conj(self):
        real = self.real
        imag = - self.imag
        return ComplexTensor(real=real, imag=imag)

    def abs(self, square=False):
        absolute_value = self.real**2 + self.imag**2
        if square is False:
            absolute_value = torch.sqrt(absolute_value)
        return absolute_value

    def __add__(self, other):
        real = self.real + other.real
        imag = self.imag + other.imag
        return ComplexTensor(real=real, imag=imag)

    def __sub__(self, other):
        real = self.real - other.real
        imag = self.imag - other.imag
        return ComplexTensor(real=real, imag=imag)

    def __mul__(self, other):
        real = self.real * other.real - self.imag * other.imag
        imag = self.real * other.imag +self.imag * other.real
        return ComplexTensor(real=real, imag=imag)

    def __rmul__(self, other):
        real = self.real * other
        imag = self.imag * other
        return ComplexTensor(real=real, imag=imag)

    def __truediv__(self, other):
        return self * other.reciprocal()

    def reciprocal(self):
        """ Computes the reciprocal of the complex number"""
        norm = self.abs(square=True)
        return ComplexTensor(real=self.real / norm, imag=-self.imag / norm)

    def numpy(self):
        """Converts the array to a numpy complex array"""
        real = self.real.numpy()
        imag = self.imag.numpy()
        return real + 1j * imag

    def stack(self, dim=-1):
        """
        Stacks the real and imaginary part to make a tensor of size (*, 2), with the real and imaginary part in
        the last two dimensions. This is useful if you need to pass this to a pytorch tensor which accepts complex
        inputs, like an ifft.
        """
        return torch.stack((self.real, self.imag), dim=dim)

    def __str__(self):
        Complex = namedtuple("ComplexTensor", ["real", "imag"])
        c = Complex(real=self.real, imag=self.imag)
        return str(c)

    def __getitem__(self, item):
        real = self.real[item]
        imag = self.imag[item]
        return ComplexTensor(real=real, imag=imag)

    def __setitem__(self, key, value):
        self.real[key] = value.real
        self.imag[key] = value.imag


    def shape(self):
        """
        Shape of the tensor. Notice that the 2D nature of complex numbers is not considered: 1+1j has dimension 1,
        not (1,2)
        """
        return self.real.shape

    def unsqueeze(self, dim):
        real = self.real.unsqueeze(dim)
        imag = self.imag.unsqueeze(dim)
        return ComplexTensor(real=real, imag=imag)

    def __rmatmul__(self, other):

        if type(other) == ComplexTensor:
            real = torch.matmul(other.real, self.real) - torch.matmul(other.imag, self.imag)
            imag = torch.matmul(other.real, self.imag) + torch.matmul(other.imag, self.real)
        else:
            real = torch.matmul(other, self.real)
            imag =  torch.matmul(other, self.imag)

        return ComplexTensor(real=real, imag=imag)


    def __matmul__(self, other):

        if type(other) == ComplexTensor:
            real = torch.matmul(self.real, other.real) - torch.matmul(self.imag, other.imag)
            imag = torch.matmul(self.real, other.imag) + torch.matmul(self.imag, other.real)
        else:
            real = torch.matmul(self.real, other)
            imag = torch.matmul(self.imag, other)

        return ComplexTensor(real=real, imag=imag)

    def transpose(self):
        real = self.real.T
        imag = self.imag.T
        return ComplexTensor(real=real, imag=imag)

    def batch_elementwise(self, v):
        """
        Computes the batch elementwise multiplication between two complex vectors.

        Example: assume M of shape [32, 200] and v of shape [200].
        The operation multiplies each row by v elementwise.

        NOTE: this can definitely be improved, for the moment I use a simple but slow for loop.

        Parameters
        ----------
        vector: ComplexTensor,
            vector of complex numbers.

        Returns
        -------
        prod: ComplexTensor,
            element wise multiplication between self and vector
        """
        prod = ComplexTensor(real=torch.zeros(self.shape()), imag=torch.zeros(self.shape()))

        for row in range(self.shape()[0]):
            prod[row] = self[row] * v
        return prod