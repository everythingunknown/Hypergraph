using Graphs
using SparseArrays
using PyCall
using LinearAlgebra
using Distributions
using Printf
using Mmap
using DataStructures
using Base.Threads, Random

# FIFO-Queue
function compute_queue_size(num_of_vertices::Int)::Int
    return 1 << ceil(Int, log2(num_of_vertices + 2))
end
mutable struct MyQueue
    mask::Int
    queue::Vector{Int}
    front::Int
    rear::Int
    num::Int

    function MyQueue(num_of_vertices::Int)
        mask = compute_queue_size(num_of_vertices) - 1
        queue = zeros(Int, mask + 2)
        new(mask, queue, 0, 0,0)
    end
end
@inline size_q(queue::MyQueue) = queue.num
@inline isempty_q(queue::MyQueue) = queue.rear == queue.front
@inline function pop_q!(queue::MyQueue)
    queue.front += 1
    elem = queue.queue[queue.front]
    queue.front &= queue.mask
    queue.num -=1
    return elem
end
@inline function push_q!(queue::MyQueue, elem::Int)
    queue.rear += 1
    queue.queue[queue.rear] = elem
    queue.rear &= queue.mask
    queue.num +=1
end
# Hypergraph Reading
function read_hypergraph_optimized(nverts_file::String, simplices_file::String)
    t1 = time()
    parse_ints(filename) = parse.(Int, split(String(Mmap.mmap(filename))))
    
    edge_sizes = parse_ints(nverts_file)
    simplices = parse_ints(simplices_file)
    
    m = length(edge_sizes)
    n = maximum(simplices)

    rows = Int[]
    cols = Int[]
    sizehint!(rows, length(simplices))
    sizehint!(cols, length(simplices))
    
    cum_sizes = cumsum(vcat(1, edge_sizes))
    
    Threads.@threads for i in 1:m
        range = cum_sizes[i]:(cum_sizes[i+1]-1)
        for j in range
            node_id = simplices[j]
            push!(rows, i)
            push!(cols, node_id)
        end
    end
    
    R = sparse(rows, cols, 1, m, n)
    W = sparse(cols, rows, 1, n, m)

    DW, invDW = sparse_rowsum_diagonal(W)
    DR, invDR = sparse_rowsum_diagonal(R)
    barW = invDW*W
    barR = invDR*R

    t2 = time()
    println("m = $m, n = $n, graph initialized! ($(t2-t1)s), sum(W) = $(sum(W))")
    
    return m, n, barW, barR,W,R,invDW,invDR
end
function sparse_rowsum_diagonal(spmat::SparseMatrixCSC)
    row_sums = sum(spmat, dims=2) 
    if minimum(row_sums) == 0
        println("Error! Zero row sum!")
    end
    D = Diagonal(vec(row_sums))  
    row_sums = 1.0 ./ row_sums
    Dinv = Diagonal(vec(row_sums))
    return D, Dinv
end
# Resistance Parameters
function getAlpha(n::Int, distribution::Int, min::Float64, max::Float64)
    if distribution == 1
        values = rand(Uniform(min, max), n)
    elseif distribution == 2
        mean_value = (min + max) / 2
        values = rand(Normal(mean_value, 1), n)
        values = clamp.(values, min, max)
    elseif distribution == 3
        alpha = 1.5
        u = rand(n)
        C1 = max^(1 - alpha) - min^(1 - alpha)
        values = (u .* C1 .+ min^(1 - alpha)) .^ (1 / (1 - alpha))
        values = clamp.(values, min, max)
    else
        throw(ArgumentError("Invalid distribution type. Use 1 for uniform or 2 for normal."))
    end
    println("alpha_min = $(mean(values))")
    return Diagonal(values)
end
# Innate Opinion
function getS(n::Int, distribution::Int, min::Float64, max::Float64)
    if distribution == 1
        values = rand(Uniform(min, max), n)
    elseif distribution == 2
        mean_value = (min + max) / 2
        values = rand(Normal(mean_value, 1), n)
        values = clamp.(values, min, max)
    elseif distribution == 3
        λ = 1.0 / (max - min)
        values = rand(Exponential(1/λ), n) .+ min  
        values = clamp.(values, min, max)
    else
        throw(ArgumentError("Invalid distribution type. Use 1 for uniform, 2 for normal, or 3 for exponential."))
    end
    max_value = maximum(values)
    min_value = minimum(values)
    mean_value = mean(values)
    println("Innate Opinion stats - Max: $max_value, Min: $min_value, Mean: $mean_value")
    return values 
end
# main experiment
function main()
    alpha_min = 0.01
    #m, n, barW, barR, W, R, invDW, invDR = read_hypergraph_optimized("./network/threads-math-sx-nverts.txt", "./network/cleaned_threads-math-sx-simplices.txt")
    #m, n, barW, barR, W, R, invDW, invDR = read_hypergraph_optimized("./network/coauth-DBLP-full-nverts.txt", "./network/cleaned_coauth-DBLP-full-simplices.txt")
    #m, n, barW, barR, W, R, invDW, invDR = read_hypergraph_optimized("./network/coauth-MAG-Geology-full-nverts.txt", "./network/cleaned_coauth-MAG-Geology-full-simplices.txt")
    m, n, barW, barR, W, R, invDW, invDR = read_hypergraph_optimized("./network/coauth-MAG-History-full-nverts.txt", "./network/cleaned_coauth-MAG-History-full-simplices.txt")
    #m, n, barW, barR, W, R, invDW, invDR = read_hypergraph_optimized("./network/threads-stack-overflow-full-nverts.txt", "./network/cleaned_threads-stack-overflow-full-simplices.txt")
    #m, n, barW, barR, W, R, invDW, invDR = read_hypergraph_optimized("./network/threads-ask-ubuntu-nverts.txt", "./network/cleaned_threads-ask-ubuntu-simplices.txt")
    #m, n, barW, barR, W, R, invDW, invDR = read_hypergraph_optimized("./network/email-Eu-full-nverts.txt", "./network/cleaned_email-Eu-full-simplices.txt")
    #m, n, barW, barR, W, R, invDW, invDR = read_hypergraph_optimized("./network/email-Enron-full-nverts.txt", "./network/cleaned_email-Enron-full-simplices.txt")
    #m, n, barW, barR, W, R, invDW, invDR = read_hypergraph_optimized("./network/congress-bills-full-nverts.txt", "./network/cleaned_congress-bills-full-simplices.txt")
    #m, n, barW, barR, W, R, invDW, invDR = read_hypergraph_optimized("./network/contact-high-school-nverts.txt", "./network/cleaned_contact-high-school-simplices.txt")
    P = barW*barR
    row_sum_p = sum(P, dims=2) 
    println("rowsum_P_max = $(maximum(row_sum_p)), rowsum_P_min = $(minimum(row_sum_p))")
    Da = getAlpha(n,1,alpha_min,1-alpha_min)
    s = getS(n,1,0.0,1.0)
    loop1,t1,z = PowerIteration(n,m,barW,barR,Da,s,1e-8,alpha_min)
    W,u,e0,B1_diag = ChebyshevIterationInitiate(n,m,W,R,invDW,invDR,Da,s,alpha_min)
    println("Initial error e0 = $e0")
    loop3,t3,z3 = ChebyIteration(n,m+n,W,u,150,alpha_min,e0,B1_diag,1e-8)
    max_iter_cg = find_k_CG(1e-8,m,n,alpha_min)
    loop4,t4,z4 = CG(n,m+n,W,u,alpha_min,1e-8,B1_diag,ceil(Int,max_iter_cg))
    loop2,t2,z2 = AsynchronousIteration(n,m,barW,barR,Da,s,1e-8,alpha_min)
    println("sum(s) = $(sum(s)) ,sum(z) = $(sum(z)),sum(z2) = $(sum(z2)),sum(z3) = $(sum(z3))")
    println("sum(z4) = $(sum(z4))")
    t0 = 0
    if(n < 30000)
        t0,z0 = Accurate(n,m,barW,barR,Da,s)
        println("sum(z0) = $(sum(z0))")
    end
    sum_e = sum(R)
    linf1 = l21 = linf2 = l22 = linf3 = l23 = linf4 = l24 = "-"
    if n <= 30000
        linf1 = norm(z0 - z, Inf)
        l21   = norm(z0 - z, 2)

        linf2 = norm(z0 - z2, Inf)
        l22   = norm(z0 - z2, 2)

        linf3 = norm(z0 - z3, Inf)
        l23   = norm(z0 - z3, 2)

        linf4 = norm(z0 - z4, Inf)
        l24   = norm(z0 - z4, 2)
    end
    open("data.txt", "a") do io
        println(io, "$n,$m,$sum_e,$t0,$t1,$loop1,$linf1,$l21,$t2,$loop2,$linf2,$l22,$t3,$loop3,$linf3,$l23,$t4,$loop4,$linf4,$l24")
    end
end
# Exact
function Accurate(n::Int, m::Int, barW::SparseMatrixCSC, barR::SparseMatrixCSC, Da::Diagonal, s::Vector)
    t1 = time()
    W = zeros(n,n)
    W = I(n)-(I(n) - Da)*barW*barR
    invW = inv(Matrix(W))
    z = invW*Da*s
    t2 = time()
    println("Accurate Inversion time cost = $(t2-t1)s")
    return (t2-t1), z
end
# SynIt
function PowerIteration(n::Int,m::Int, barW::SparseMatrixCSC, barR::SparseMatrixCSC,
                       Da::Diagonal, s::Vector, epsilon::Float64, alpha_min::Float64;
                       max_iters::Int=3000)
    t1 = time()
    I_n = Diagonal(ones(n))  
    Alpha = I_n - Da         
    Da_s = Da * s            
    z = zeros(n)
    z_temp = zeros(m)
    converged = false
    t = 1
    while t <= max_iters
        mul!(z_temp, barR, z)
        mul!(z, barW, z_temp)
        z .= Alpha * z .+ Da_s  
        (1 - alpha_min)^t <= epsilon && (converged = true; break)
        t += 1
    end
    t2 = time()
    println("Synchronous iteration = $t, time = $(t2-t1)")
    converged || @warn "Not converged in $max_iters error $((1 - alpha_min)^t)"
    return t,(t2-t1),z
end
# AsyIt
function AsynchronousIteration(n::Int, m::Int, barW::SparseMatrixCSC, barR::SparseMatrixCSC,
    Da::Diagonal, s::Vector, epsilon::Float64, alpha_min::Float64) 
    t1 = time()
    t = 0
    Q1 = MyQueue(n)
    Q2 = MyQueue(m)
    mark1 = trues(n)
    mark2 = falses(m)
    active_nodes = Set{Int}()
    active_edges = Set{Int}()
    z = zeros(n)
    r1 = Da * s  
    r2 = zeros(m)
    for i in 1:n
        push_q!(Q1,i)
    end
    while !isempty_q(Q1)
        t += 1
        while !isempty_q(Q1)
            v = pop_q!(Q1)
            mark1[v] = false
            z[v] += r1[v]
            col_start = barR.colptr[v]
            col_end = barR.colptr[v+1]-1
            
            @simd for idx in col_start:col_end
                e = barR.rowval[idx]
                val = barR.nzval[idx]  
                r2[e] += val * r1[v]   
                if !mark2[e]
                    push_q!(Q2,e)
                    mark2[e] = true
                end
            end
            
            r1[v] = 0.0
        end
        while !isempty_q(Q2)
            e = pop_q!(Q2)
            mark2[e] = false   
            row_start = barW.colptr[e]
            row_end = barW.colptr[e+1]-1
            
            @simd for idx in row_start:row_end
                v = barW.rowval[idx]
                val = barW.nzval[idx]
                delta = (1 - Da.diag[v]) * val * r2[e]
                r1[v] += delta
                if (r1[v] > Da.diag[v] * epsilon) && !mark1[v]
                    push_q!(Q1,v)
                    mark1[v] = true
                end
            end
            r2[e] = 0.0
        end
    end
    t2 = time()
    println("AsynchronousIteration $t, time cost $(t2-t1)s")
    return t,(t2-t1),z
end
# ChebyIt
function ChebyshevIterationInitiate(n::Int, m::Int, W::SparseMatrixCSC, R::SparseMatrixCSC,
                           invDW::Diagonal, invDR::Diagonal, Da::Diagonal,
                           s::Vector, alpha_min::Float64)
    N = n + m
    A = vcat(
        hcat(spzeros(n, n), W),
        hcat(R, spzeros(m, m))
    ) |> dropzeros
    D_inv = Diagonal(vcat(diag(invDW), diag(invDR)))
    D = Diagonal(1.0 ./ diag(D_inv))
    Da_ext = Diagonal(vcat(diag(Da), zeros(m)))
    s_ext = vcat(s, zeros(m))
    diag_I_minus_Da = diag(I(N) - Da_ext)
    B1_diag = Vector(sqrt.(diag_I_minus_Da .* diag(D_inv)))
    B2_diag = Vector(sqrt.(diag(D) .* (1.0 ./ diag_I_minus_Da)))
    B2 = Diagonal(B2_diag)
    B1 = Diagonal(B1_diag)
    W = B1*A
    W = W*B1
    u = Vector(B2*(Da_ext*s_ext))
    e0 = sqrt(sum(D*Diagonal(1.0 ./ diag(I(N)-Da_ext))))
    return W,u,e0,diag(B1)
end
function truncate_vector(z::AbstractVector, n::Int)
    if issparse(z)
        nzind = z.nzind
        nzval = z.nzval
        mask = nzind .<= n
        SparseVector(n, nzind[mask], nzval[mask])
    else
        @view z[1:n]
    end
end
function ChebyIteration(n0::Int, n::Int,H::SparseMatrixCSC{Float64, Int64},c::Vector{Float64},limit::Int,
    alpha::Float64,e::Float64,B1::Vector{Float64},epsilon::Float64)
    alpha_min = alpha
    w_1 = 1.0 / sqrt(1-alpha)
    y_prev = zeros(n)
    y_next = zeros(n)
    z = zeros(n)
    y_curr = c
    zeta_prev = 1.0
    zeta_curr = 1.0 / sqrt(1.0 - alpha_min)
    zeta_next = 0.0
    e_curr = e
    i = 0
    b = w_1 + sqrt(w_1 * w_1 - 1)
    B = Diagonal(B1)
    t1 = time()
    while e_curr > epsilon
        i += 1
        zeta_next = 2/sqrt(1-alpha_min)*zeta_curr - zeta_prev
        y_next = 2*zeta_curr / sqrt(1-alpha_min) / zeta_next * (H*y_curr+c) - zeta_prev/zeta_next * y_prev
        zeta_prev = copy(zeta_curr)
        zeta_curr = copy(zeta_next)
        y_prev = copy(y_curr)
        y_curr = copy(y_next)
        e_curr = e*2.0/((b^i)+(b^(-i)))        
    end
    t2 = time()
    t3 = t2-t1
    z = B*y_curr
    z0 = truncate_vector(z,n0)
    return i,t3,z0
end
# CG
function CG(n0::Int, N::Int, W::SparseMatrixCSC{Float64,Int64}, 
    b::Vector{Float64}, alpha::Float64, 
    epsilon::Float64, B1::Vector{Float64},max_iter::Int)
    t1 = time()
    Ad = similar(b)  
    Wd = similar(b)   
    r = similar(b)    
    r_prev = similar(b) 
    x = zeros(N)
    e0 = 1.0 / (1 - sqrt(1-alpha))
    mul!(Wd, W, x)     
    @. Ad = x - Wd     
    @. r = b - Ad      
    d = copy(r)        
    iter = 0
    converged = false

    while iter < max_iter && !converged
        mul!(Wd, W, d)  
        @. Ad = d - Wd  
        α = dot(r, r) / dot(d, Ad)

        @. x += α * d    
        @. r_prev = r    
        @. r -= α * Ad   

        res_norm = norm(r,2) * e0
        converged = res_norm < epsilon  

        β = dot(r, r) / dot(r_prev, r_prev)
        @. d = r + β * d  

        iter += 1
    end

    @. x *= B1  

    z0 = truncate_vector(x, n0)
    t2 = time()
    println("CG converged in $iter iterations, time: $(round(t2-t1,digits=3))s")
    return iter, (t2-t1),z0
end
function find_k_CG(epsilon::Float64,m::Int,n::Int,alpha_min::Float64)
    phi = sqrt((1.0+sqrt(1.0-alpha_min))/(1.0 - sqrt(1.0-alpha_min)))
    k = log2(epsilon / 2 / phi / (m+n)) / log2((phi-1)/(phi+1))
    println("k = $k")
    return k
end
# Small network experiment
function smallNetworkResidue()
    #m, n, barW, barR, W, R, invDW, invDR = read_hypergraph_optimized("./network/email-Enron-full-nverts.txt", "./network/cleaned_email-Enron-full-simplices.txt")
    #m, n, barW, barR, W, R, invDW, invDR = read_hypergraph_optimized("./network/contact-high-school-nverts.txt", "./network/cleaned_contact-high-school-simplices.txt")
    #m, n, barW, barR, W, R, invDW, invDR = read_hypergraph_optimized("./network/email-Eu-full-nverts.txt", "./network/cleaned_email-Eu-full-simplices.txt")
    m, n, barW, barR, W, R, invDW, invDR = read_hypergraph_optimized("./network/congress-bills-full-nverts.txt", "./network/cleaned_congress-bills-full-simplices.txt")
    alpha_min = 0.1
    Da = getAlpha(n,3,alpha_min,1-alpha_min)
    s = getS(n,1,0.0,1.0)
    W,u,e0,B1_diag = ChebyshevIterationInitiate(n,m,W,R,invDW,invDR,Da,s,alpha_min)
    t0,z0 = Accurate(n,m,barW,barR,Da,s)
    R_CG(n,m+n,W,z0,u,alpha_min,1e-150,B1_diag)
    R_PowerIteration(n,m,barW,barR,Da,s,1e-150,alpha_min,z0)
    R_AsynchronousIteration(n,m,barW,barR,Da,s,1e-150,alpha_min,z0)
    R_ChebyIteration(n,m+n,W,u,150,alpha_min,e0,B1_diag,1e-150,z0)
end
function R_ChebyIteration(n0::Int, n::Int,H::SparseMatrixCSC{Float64, Int64},c::Vector{Float64},limit::Int,
    alpha::Float64,e::Float64,B1::Vector{Float64},epsilon::Float64,z0::Vector)
    alpha_min = alpha
    w_1 = 1.0 / sqrt(1-alpha)
    y_prev = zeros(n)
    y_next = zeros(n)
    z = zeros(n)
    y_curr = c
    zeta_prev = 1.0
    zeta_curr = 1.0 / sqrt(1.0 - alpha_min)
    zeta_next = 0.0
    e_curr = e
    i = 0
    b = w_1 + sqrt(w_1 * w_1 - 1)
    B = Diagonal(B1)
    t1 = time()
    while i <= 101
        if i%10 == 0 && i>0
            z = B*y_curr
            z1 = truncate_vector(z,n0)
            norm2 = norm((z0-z1),2)
            open("small.txt", "a") do io
                println(io,"Cheby $i $norm2")
            end
        end
        i += 1
        zeta_next = 2/sqrt(1-alpha_min)*zeta_curr - zeta_prev
        y_next = 2*zeta_curr / sqrt(1-alpha_min) / zeta_next * (H*y_curr+c) - zeta_prev/zeta_next * y_prev
        zeta_prev = copy(zeta_curr)
        zeta_curr = copy(zeta_next)
        y_prev = copy(y_curr)
        y_curr = copy(y_next)
        e_curr = e*2.0/((b^i)+(b^(-i)))        
    end
    t2 = time()
    t3 = t2-t1
    z = B*y_curr
    z0 = truncate_vector(z,n0)
    return i,t3,z0
end
function R_AsynchronousIteration(n::Int, m::Int, barW::SparseMatrixCSC, barR::SparseMatrixCSC,
    Da::Diagonal, s::Vector, epsilon::Float64, alpha_min::Float64, z0::Vector)
    
    t1 = time()
    t = 0
    Q1 = MyQueue(n)
    Q2 = MyQueue(m)
    mark1 = trues(n)
    mark2 = falses(m)
    active_nodes = Set{Int}()
    active_edges = Set{Int}()
    z = zeros(n)
    r1 = Da * s  
    r2 = zeros(m)
    for i in 1:n
        push_q!(Q1,i)
    end
    while !isempty_q(Q1)
        if(t % 10 ==0 && t > 0)
            norm2 = norm((z0-z),2)
            open("small.txt", "a") do io
                println(io,"AsyIt $t $norm2")
            end
        end
        t += 1
        if t >= 101 
            break
        end
        while !isempty_q(Q1)
            v = pop_q!(Q1)
            mark1[v] = false
            z[v] += r1[v]
            col_start = barR.colptr[v]
            col_end = barR.colptr[v+1]-1
            
            @simd for idx in col_start:col_end
                e = barR.rowval[idx]
                val = barR.nzval[idx]  
                r2[e] += val * r1[v]   
                if !mark2[e]
                    push_q!(Q2,e)
                    mark2[e] = true
                end
            end
            
            r1[v] = 0.0
        end
        
        while !isempty_q(Q2)
            e = pop_q!(Q2)  
            mark2[e] = false   
            row_start = barW.colptr[e]
            row_end = barW.colptr[e+1]-1
            
            @simd for idx in row_start:row_end
                v = barW.rowval[idx]
                val = barW.nzval[idx]
                delta = (1 - Da.diag[v]) * val * r2[e]
                r1[v] += delta
                if (r1[v] > Da.diag[v] * epsilon) && !mark1[v]
                    push_q!(Q1,v)
                    mark1[v] = true
                end
            end
            r2[e] = 0.0
        end
    end
    t2 = time()
    println("AsynchronousIteration $t, time cost $(t2-t1)s")
    return t,(t2-t1),z
end
function R_PowerIteration(n::Int,m::Int, barW::SparseMatrixCSC, barR::SparseMatrixCSC,
    Da::Diagonal, s::Vector, epsilon::Float64, alpha_min::Float64,z0 ::Vector ;
    max_iters::Int=101 )
    t1 = time()
    I_n = Diagonal(ones(n))  
    Alpha = I_n - Da         
    Da_s = Da * s            
    z = zeros(n)
    z_temp = zeros(m)
    converged = false
    t = 1
    while t <= max_iters
        mul!(z_temp, barR, z)
        mul!(z, barW, z_temp)
        z .= Alpha * z .+ Da_s  
        t += 1
        if (t % 10 == 0)
            norm2 = norm((z0-z),2)
            open("small.txt", "a") do io
                println(io,"SynIt $t $norm2")
            end
        end
    end
    t2 = time()
    println("Synchronous iteration = $t, time = $(t2-t1)")
    return t,(t2-t1),z
end
function R_CG(n0::Int, N::Int, W::SparseMatrixCSC{Float64,Int64}, z0::Vector{Float64},
    b::Vector{Float64}, alpha::Float64, epsilon::Float64, B1::Vector{Float64})
    t1 = time()
    Ad = similar(b)  
    Wd = similar(b)   
    r = similar(b)    
    r_prev = similar(b) 
    x = zeros(N)
    e0 = 1.0 / (1 - sqrt(1-alpha))

    mul!(Wd, W, x)     
    @. Ad = x - Wd     
    @. r = b - Ad     

    d = copy(r)        
    iter = 0
    max_iter = 101
    converged = false

    while iter < max_iter
        mul!(Wd, W, d)  
        @. Ad = d - Wd  

        α = dot(r, r) / dot(d, Ad)

        @. x += α * d    
        @. r_prev = r    
        @. r -= α * Ad  

        β = dot(r, r) / dot(r_prev, r_prev)
        @. d = r + β * d  
        iter += 1

        if iter % 10 ==0
            tempz = @. x*B1
            z = truncate_vector(tempz,n0)
            norm_2 = norm((z-z0),2)
            open("small.txt", "a") do io
                println(io,"CG $iter $norm_2")
            end
        end
    end

    t2 = time()
    println("CG converged in $iter iterations, time: $(round(t2-t1,digits=3))s")
    return iter, (t2-t1),z0
end

main()
#smallNetworkResidue()