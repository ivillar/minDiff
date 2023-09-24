## minDiff

A minimalistic automatic differentiation library based on the Pytorch API.
Backed by NumPy.

TODOs:
1. Subclass np.ndarray for Tensor class
    1. Add indexing functionality to Tensor class
2. Add Conv2D layer support
3. Add missing docstrings
4. Make it so that calling a tensor feels more like how PyTorch does it (ex. md.Tensor instead of md.ts.Tensor)
5. Add DataLoader support
6. Add BatchNorm and Dropout