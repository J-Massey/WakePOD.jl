import numpy as np
import scipy.sparse as sp


def create_grad_operator_y(nx, ny, dy):
    size = nx * ny

    # Diagonal values
    main_diag = np.ones(size) * 0
    off_diag = np.ones(size) * -0.5
    off_diag2 = np.ones(size) * 0.5

    # Construct the base operator
    diagonals = [main_diag, off_diag, off_diag2]
    offsets = [0, -1, 1]
    grad_operator = sp.diags(diagonals, offsets, shape=(size, size), format="lil")

    for idx in range(nx):
        grad_operator[idx * ny, idx * ny - 1] = 0
        grad_operator[idx * ny, idx * ny] = -1.5
        grad_operator[idx * ny, idx * ny + 1] = 2
        grad_operator[idx * ny, idx * ny + 2] = -0.5
        grad_operator[-idx * (ny) - 1, -idx * (ny)] = 0
        grad_operator[-idx * (ny) - 1, -idx * (ny) - 1] = 1.5
        grad_operator[-idx * (ny) - 1, -idx * (ny) - 2] = -2
        grad_operator[-idx * (ny) - 1, -idx * (ny) - 3] = 0.5

    return grad_operator.tocsr() / dy


def create_grad_operator_x(nx, ny, dx):
    size = nx * ny
    # Diagonal values
    main_diag = np.ones(size) * 0
    off_diag = np.ones(size) * -0.5
    off_diag2 = np.ones(size) * 0.5
    diagonals = [main_diag, off_diag, off_diag2]
    # Offsets for diagonals
    offsets = [0, -ny, ny]
    grad_operator = sp.diags(diagonals, offsets, shape=(size, size), format="lil")
    for idx in range(ny):
        grad_operator[idx] = 0
        grad_operator[idx, idx] = -1.5
        grad_operator[idx, ny + idx] = 2
        grad_operator[idx, 2 * ny + idx] = -0.5
        bc = size - 1
        grad_operator[bc - idx] = 0
        grad_operator[bc - idx, bc - idx] = 1.5
        grad_operator[bc - idx, bc - idx - ny] = -2
        grad_operator[bc - idx, bc - idx - 2 * ny] = 0.5
    return grad_operator.tocsr() / dx


def create_laplacian_operator_x(nx, ny, dx=1):
    size = nx * ny
    # Diagonal values
    main_diag = np.ones(size) * -2
    off_diag = np.ones(size) * 1
    off_diag2 = np.ones(size) * 1
    diagonals = [main_diag, off_diag, off_diag2]
    # Offsets for diagonals
    offsets = [0, -ny, ny]
    laplacian_operator = sp.diags(diagonals, offsets, shape=(size, size), format="lil")
    for idx in range(ny):
        laplacian_operator[idx] = 0
        laplacian_operator[idx, idx] = 2
        laplacian_operator[idx, ny + idx] = -5
        laplacian_operator[idx, 2 * ny + idx] = 4
        laplacian_operator[idx, 3 * ny + idx] = -1
        bc = size - 1
        laplacian_operator[bc - idx] = 0
        laplacian_operator[bc - idx, bc - idx] = 2
        laplacian_operator[bc - idx, bc - idx - ny] = -5
        laplacian_operator[bc - idx, bc - idx - 2 * ny] = 4
        laplacian_operator[bc - idx, bc - idx - 3 * ny] = -1
    return laplacian_operator.tocsr() / (dx**2)


def create_laplacian_operator_y(nx, ny, dy):
    size = nx * ny
    # Diagonal values
    main_diag = np.ones(size) * -2
    off_diag = np.ones(size) * 1
    off_diag2 = np.ones(size) * 1
    diagonals = [main_diag, off_diag, off_diag2]
    # Offsets for diagonals
    offsets = [0, -1, 1]
    grad_operator = sp.diags(diagonals, offsets, shape=(size, size), format="lil")
    for idx in range(nx):
        grad_operator[idx * (ny), idx * (ny) - 1] = 0
        grad_operator[idx * (ny), idx * (ny)] = 2
        grad_operator[idx * (ny), idx * (ny) + 1] = -5
        grad_operator[idx * (ny), idx * (ny) + 2] = 4
        grad_operator[idx * (ny), idx * (ny) + 3] = -1
        grad_operator[-idx * (ny) - 1, -idx * (ny)] = 0
        grad_operator[-idx * (ny) - 1, -idx * (ny) - 1] = 2
        grad_operator[-idx * (ny) - 1, -idx * (ny) - 2] = -5
        grad_operator[-idx * (ny) - 1, -idx * (ny) - 3] = 4
        grad_operator[-idx * (ny) - 1, -idx * (ny) - 4] = -1
    return grad_operator.tocsr() / (dy**2)
