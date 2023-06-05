using LinearAlgebra
using Test
"""
    intuitiveSVD(A)

Computes the singular value decomposition (SVD) of a matrix using the intuitive method.

# Arguments
- `A`: Data matrix of shape (m, n) representing the data.

"""
function intuitiveSVD(A)
    U, Σ, V = svd(A)
    return U, Σ, V
end
"""
    truncatedSVD(A, k)

Computes the truncated singular value decomposition (SVD) of a matrix using the intuitive method.

# Arguments
- `A`: Data matrix of shape (m, n) representing the data.
- `k`: Number of desired singular values/vectors.
"""
function truncatedSVD(A, k)
    U, Σ, V = svd(A)
    Uk = U[:, 1:k]
    Σk = Σ[1:k]
    Vk = V[:, 1:k]
    return Uk, Σk, Vk
end
"""
    qrSVD(A,k)

Computes the truncated SVD using the QR decomposition.

# Arguments
- `A`: Data matrix of shape (m, n) representing the data.
- `k`: Number of desired singular values/vectors.
"""
function qrSVD(A,k)
    Q,R = qr(A)
    B=transpose(Matrix(Q))*A
    Û,Σ,Vt=svd(B)
    max_idx = sortperm(Σ, rev=true)[1:k]
    Σk = Σ[max_idx]
    Uk=Û[:, max_idx]
    Vk=Vt'[max_idx, :]

    U=Q*Uk
    return U,Σk,Vk
end

"""
    rSVD(A,k)

Computes the SVD using the randomised method.

# Arguments
- `A`: Data matrix of shape (m, n) representing the data.
- `k`: Number of desired singular values/vectors.
"""
function randomSVD(A, k)
    (m,n)=size(A)
    Φ=rand(n,k)
    Ar=A*Φ
    Q,R = qr(Ar)
    B=transpose(Matrix(Q))*A
    Û,Σ,Vt=svd(B)
    max_idx = sortperm(Σ, rev=true)[1:k]
    Σk = Σ[max_idx]
    Uk=Û[:, max_idx]
    Vk=Vt'[max_idx, :]
    U=Matrix(Q)*Uk
    return U,Σk,Vk
end
"""
    initialize(A, k)

Initializes the data matrix and computes the initial left singular vectors and singular values.

# Arguments
- `A`: Data matrix of shape (m, n) representing the first batch of data.
- `k`: Number of desired singular values/vectors.

# Returns
- `U`: Initial matrix of shape (m, k) containing the left singular vectors.
"""
function initialize(A, k)
    Q,R = qr(A)
    B=transpose(Matrix(Q))*A
    Û,Σ,Vt=svd(B)

    max_idx = sortperm(Σ, rev=true)[1:k]
    Uk = Û[:, max_idx]
    U=Matrix(Q)*Uk
    return U
end

"""
    incorporate_data(A, M, Σ, k)

Incorporates new data into the existing data matrix using singular value decomposition (SVD).

# Arguments
- `A`: Existing data matrix of shape (m, n).
- `U`: Matrix of shape (m, k) containing the left singular vectors.
- `k`: Number of singular values to incorporate.

# Returns
- `M`: Updated matrix of shape (m, k) containing the left singular vectors.
- `Σ`: Updated array of length k containing the singular values.
"""

function incorporate_data(A, U, k)
    m_ap = hcat(U, A)
    Qi, Ri = qr(m_ap)
    Bi=transpose(Matrix(Qi))*A
    Ûi,Σi,Vti = svd(Bi)

    max_idx = sortperm(Σi, rev=true)[1:k]
    Σi = Σi[max_idx]
    Uki = Ûi[:, max_idx]
    Ui=Matrix(Qi)*Uki
    Vki=Vti'[max_idx, :]
    return Ui, Σi, Vki
end
"""
    reconstruction(U, Σ, V)

Reconstructs the data matrix from the singular value decomposition (SVD).
"""
function reconstruction(Uk, Σk, Vk, start_mode=1, end_mode=1)
    Ak = diagm(Σk) * Vk
    M = Uk[:,start_mode:end_mode] * Ak[start_mode:end_mode, :]
    return M
end

@testset "Test the SVDs" begin
    m=1000; n=50; k=10
    A=rand(m,n)
    @testset "Streaming parts" begin
        U = initialize(A, k)
        @test size(U) == (size(A, 1), k)
        A2 = rand(m, n)
        U, Σ, V = incorporate_data(A2, U, k)
        @test size(U) == (size(A, 1), k)
        @test length(Σ) == k
    end
    @testset "intuitive" begin
        U, Σ, V = truncatedSVD(A, k)
        @test size(U) == (size(A, 1), k)
        @test length(Σ) == k
    end
    @testset "qr" begin
        U, Σ, V = qrSVD(A, k)
        @test size(U) == (size(A, 1), k)
        @test length(Σ) == k
    end
    @testset "random" begin
        U, Σ, V = randomSVD(A, k)
        @test size(U) == (size(A, 1), k)
        @test length(Σ) == k
    end
end