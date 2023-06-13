include("SVD.jl")
using NPZ
using GLMakie
using LinearAlgebra
using Statistics

xlims, ylims = (-0.35, 2), (-0.35, 0.35)

flow_snapshots_u = npzread("data/u.npy")
flow_snapshots_u = permutedims(flow_snapshots_u, [3, 2, 1])
flow_snapshots_v = npzread("data/v.npy")
flow_snapshots_v = permutedims(flow_snapshots_v, [3, 2, 1])
# Just the wake
nx, ny, nt = size(flow_snapshots_u)
dt = 8/nt
pxs = LinRange(xlims..., nx)
pys = LinRange(ylims..., ny)

wake_snapshots_u = flow_snapshots_u[pxs.>1, :, :]
wake_snapshots_v = flow_snapshots_v[pxs.>1, :, :]
pxs = pxs[pxs.>1]
nx, ny, nt = size(wake_snapshots_u)

# take away the time mean
# flow_snapshots_u = flow_snapshots_u .- mean(flow_snapshots_u, dims = 3)
# flow_snapshots_v = flow_snapshots_v .- mean(flow_snapshots_v, dims = 3)

# get mag and vorticity
function curl(u, v, dx, dy)
    ∂v_∂x = zeros(size(v))
    ∂u_∂y = zeros(size(u))
    
    # Compute ∂v/∂x
    @views ∂v_∂x[:, 2:end-1] .= (v[:, 3:end] - v[:, 1:end-2]) / (2 * dx)
    
    # Compute ∂u/∂y
    @views ∂u_∂y[2:end-1, :] .= (u[3:end, :] - u[1:end-2, :]) / (2 * dy)
    
    return ∂v_∂x - ∂u_∂y
end

# Calculate vorticity snapshots
vorticity_snapshots = zeros(size(wake_snapshots_u))

dx = mean(pxs[2:end] - pxs[1:end-1])
dy = mean(pys[2:end] - pys[1:end-1])

for t in 1:nt
    vorticity_snapshots[:, :, t] = curl(wake_snapshots_u[:, :, t], wake_snapshots_v[:, :, t], dx, dy)
end

function testplotvort()
    # Quick test plot
    f = Figure(resolution = (1600, 400))
    ax = Axis(f[1, 1])
    co = contourf!(ax,
        pxs, pys,vorticity_snapshots[:,:,1],
        xlabel=L"x", ylabel=L"y", title=L"ω_z",
        levels = range(-0.05, 0.05, length = 44),
        colormap=:seismic,
        extendlow = :auto, extendhigh = :auto,
    )
    tightlimits!(ax)
    return f
end
f = testplotvort()
save("figures/testplotvort.png", f)

mag_snapshots = (wake_snapshots_u.^2 .+ wake_snapshots_v.^2).^0.5

function testplotmag()
    # Quick test plot
    f = Figure(resolution = (1600, 400))
    ax = Axis(f[1, 1])
    co = contourf!(ax,
        pxs, pys,mag_snapshots[:,:,1],
        xlabel=L"x", ylabel=L"y", title=L"ω_z",
        levels = range(-0.1, 1.2, length = 44),
        # colormap=:icefire,
        extendlow = :auto, extendhigh = :auto,
    )
    tightlimits!(ax)
    return f
end
f = testplotmag()
save("figures/testplotmag.png", f)


# Flatten snapshots along first two dims
flat_snapshots_u = reshape(
    wake_snapshots_u,
    nx*ny,
    nt
    )
flat_snapshots_v = reshape(
    wake_snapshots_v,
    nx*ny,
    nt
    )
flat_vorticity_snapshots = reshape(
    vorticity_snapshots,
    nx*ny,
    nt
    )

k=20
U, Σ, Vk = rSVD(flat_vorticity_snapshots, k)

# for m in 1:k
#     f = Figure(resolution = (1600, 400))
#     ax = Axis(f[1, 1])
#     ϕ = reshape(U[:, m], nx, ny)
#     println(minimum(ϕ), " ", maximum(ϕ))
#     co = contourf!(ax,
#         pxs, pys, reshape(U[:, m], nx, ny),
#         xlabel=L"x", ylabel=L"y", title=L"ϕ_$m",
#         levels = range(-0.01, 0.01, length = 44),
#         # colormap=:plasma,
#         extendlow = :auto, extendhigh = :auto,
#     )
#     tightlimits!(ax)
#     save("figures/modes/U_$m.png", f)
# end

A = diagm(Σ[1:k]) * Vk

using FileIO
using ImageMagick

# Loop over each mode
for m in 1:k
    # Create an array to store frames for the GIF
    Mk = U[:,m:m] * A[m:m, :]
    frames = []

    # Loop over each time step
    for t in 1:nt
        Mpl = reshape(Mk[:, t], nx, ny)
        fig = Figure(resolution = (1600, 400))
        ax = Axis(fig[1, 1])
        lines = contourf!(ax, pxs, pys, Mpl,
            levels = range(-0.01, 0.01, length = 44),
            colormap = :seismic,
            extend = :both,
        )
        xlabel!(ax, "x")
        ylabel!(ax, "y")
        title!(ax, "ϕ_$(m)(t)")
        tightlimits!(ax)

        # Push the current frame to the frames array
        push!(frames, frame(fig))
    end

    # Save the frames as a GIF
    save(string("figures/time/mode_", m, ".gif"), frames, fps = 10)
end