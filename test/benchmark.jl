using LinearAlgebra
using BenchmarkTools
include("../src/SVD.jl")

m=100000; n=50; k=10
# A=rand(m,n)

# b1 = @benchmarkable intuitiveSVD(y) setup=(y = copy($A))
# tune!(b1)
# run(b1)

# b2 = @benchmarkable truncatedSVD(y,5) setup=(y = copy($A))
# tune!(b2)
# run(b2)

# b3 = @benchmarkable SVDqr(y,5) setup=(y = copy($A))
# tune!(b3)
# run(b3)

# b4 = @benchmarkable rSVD(y,5) setup=(y = copy($A))
# tune!(b4)
# run(b4)

# To test the streaming we need to split up the matrix
A1=rand(m,25)
# A2=rand(m,25)

# function stream(A1, A2, k)
#     U = initialize(A1,k)
#     return incorporate(A2,U,k)
# end

# b5 = @benchmarkable stream(y1, y2 ,5) setup=(y1, y2 = copy($A1), copy($A2))
# tune!(b5)
# run(b5)


function rstream(A1, A2, k)
    rinitialize(A1,k)
    # return rincorporate(A2,U,k)
end

b6 = @benchmarkable rinitialize(A1,5)
tune!(b6)
run(b6)


# Define a parent BenchmarkGroup to contain our suite
# suite = BenchmarkGroup()

# # Add some child groups to our benchmark suite. The most relevant BenchmarkGroup constructor
# # for this case is BenchmarkGroup(tags::Vector). These tags are useful for
# # filtering benchmarks by topic, which we'll cover in a later section.
# suite["utf8"] = BenchmarkGroup(["string", "unicode"])
# suite["trig"] = BenchmarkGroup(["math", "triangles"])

# # Add some benchmarks to the "utf8" group
# teststr = join(rand('a':'d', 10^4));
# suite["utf8"]["replace"] = @benchmarkable replace($teststr, "a" => "b")
# suite["utf8"]["join"] = @benchmarkable join($teststr, $teststr)

# # Add some benchmarks to the "trig" group
# for f in (sin, cos, tan)
#     for x in (0.0, pi)
#         suite["trig"][string(f), x] = @benchmarkable $(f)($x)
#     end
# end
# suite