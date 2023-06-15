import numpy as np
from scipy.linalg import svd, sqrtm, eig

def DMD_from_POD(X, Y, dt, U, S, V, k, tol=1e-16):
    nt = X.shape[1]
    Ur = U
    Sr = np.diag(S)
    Vr = V
    A = Ur.T @ Y @ (Vr.T / Sr)
    rho, W, Wadj = eigen_dual(A, np.eye(k), True)  # Assuming eigen_dual is defined elsewhere
    Psi = Y @ (Vr / Sr) @ W
    Phi = Ur @ Wadj

    for i in range(k):
        Psi[:, i] = Psi[:, i] / np.sqrt(Psi[:, i].T @ Psi[:, i])
        Phi[:, i] = Phi[:, i] / np.sqrt(Phi[:, i].T @ Phi[:, i])
        Psi[:, i] = Psi[:, i] / (Phi[:, i].T @ Psi[:, i])

    b = np.linalg.lstsq(Psi, X[:, 0], rcond=None)[0]
    large = np.abs(b) > tol * np.max(np.abs(b))
    Psi = Psi[:, large]
    Phi = Phi[:, large]
    rho = rho[large]
    lam = np.log(rho) / dt
    b = b[large]

    return lam, Psi, Phi, b


def eigen_dual(A, Q, log_sort=False):
    Aadj = adj(A, Q)
    if log_sort:
        lam, V = eig(A, sort=lambda x: np.imag(np.log(x)))
        lam_adj, W = eig(Aadj, sort=lambda x: -np.imag(np.log(x)))
        p = np.argsort(-np.real(np.log(lam)))
        p_adj = np.argsort(-np.real(np.log(lam_adj)))
    else:
        lam, V = eig(A, sort=lambda x: np.imag(x))
        lam_adj, W = eig(Aadj, sort=lambda x: -np.imag(x))
        p = np.argsort(-np.real(lam))
        p_adj = np.argsort(-np.real(lam_adj))
    
    V = V[:, p]
    lam = lam[p]
    W = W[:, p_adj]
    lam_adj = lam_adj[p_adj]
    V = normalize_basis(V, Q)
    W = normalize_basis(W, Q)

    for i in range(V.shape[1]):
        V[:, i] = V[:, i] / (W[:, i].T @ Q @ V[:, i])

    return lam, V, W


def normalize_basis(V, Q):
    for i in range(V.shape[1]):
        V[:, i] = V[:, i] / np.sqrt(V[:, i].T @ Q @ V[:, i])
    return V


def adj(A, Q):
    Aadj = np.linalg.solve(Q, A.T @ Q)
    return Aadj


xlims, ylims = (1, 2), (-0.35, 0.35)
vort1 = np.load("data/wake_vort.npy")
vort2 = np.roll(vort1, 1, axis=2)

print("Loaded snaps")

k = 20
nx, ny, nt = vort1.shape
dt = 8 / nt
pxs = np.linspace(*xlims, nx)
pys = np.linspace(*ylims, ny)

flat_vort1 = vort1.reshape(nx*ny, nt)
flat_vort2 = vort2.reshape(nx*ny, nt)

U = np.load("data/U_POD.npy")
S = np.load("data/S_POD.npy")
Vk = np.load("data/Vk_POD.npy")
# print(U[:, :k].shape)

A = np.diag(S) @ Vk
print(A.shape)


nt = flat_vort1.shape[1]
Sr = np.diag(S)
A = (np.transpose(U) @ flat_vort2) * (np.linalg.inv(Sr) @ Vk)

print(flat_vort2.shape)
Aadj = A[1:k, 1:k]
lam, V = eig(A)#, sort=lambda x: np.imag(x))
lam_adj, W = eig(Aadj)#, sort=lambda x: -np.imag(x))
p = np.argsort(-np.real(lam))
p_adj = np.argsort(-np.real(lam_adj))

V = V[:, p]
lam = lam[p]
W = W[:, p_adj]
lam_adj = lam_adj[p_adj]
V = normalize_basis(V, Q)
W = normalize_basis(W, Q)

for i in range(V.shape[1]):
    V[:, i] = V[:, i] / (W[:, i].T @ Q @ V[:, i])
rho, W, Wadj = eigen_dual(A, np.eye(k), True)  # Assuming eigen_dual is defined elsewhere
Psi = flat_vort2 @ (Vr / Sr) @ W
Phi = Ur @ Wadj

for i in range(k):
    Psi[:, i] = Psi[:, i] / np.sqrt(Psi[:, i].T @ Psi[:, i])
    Phi[:, i] = Phi[:, i] / np.sqrt(Phi[:, i].T @ Phi[:, i])
    Psi[:, i] = Psi[:, i] / (Phi[:, i].T @ Psi[:, i])

b = np.linalg.lstsq(Psi, flat_vort1[:, 0], rcond=None)[0]
large = np.abs(b) > 1e-16 * np.max(np.abs(b))
Psi = Psi[:, large]
Phi = Phi[:, large]
rho = rho[large]
lam = np.log(rho) / dt
b = b[large]







lam, Psi, Phi, b = DMD_from_POD(flat_vort1, flat_vort2, dt, U, S, Vk, k)