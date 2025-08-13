module Strategies

    using DataFrames, Dates, Statistics, StatsBase
    export Strategy, MomentumTS, PairsCoint, Seasonality, param_grid, fit!, positions, pairs_gross_returns

    abstract type Strategy end

    # --- Momentum ---
    struct MomentumTS <: Strategy
        col_price::Symbol
        allow_short::Bool
    end
    MomentumTS(;col_price=:Close, allow_short=false) = MomentumTS(col_price, allow_short)

    function param_grid(::MomentumTS)
        return [(fast=10, slow=50), (fast=20, slow=100), (fast=50, slow=200)]
    end

    function fit!(::MomentumTS, ::DataFrame)
        return nothing
    end

    function ma(x::AbstractVector, n::Int)
        n <= 1 && return collect(x)
        out = similar(collect(x), Float64)
        s = 0.0
        for i in eachindex(x)
            s += x[i]
            if i > n
                s -= x[i-n]
                out[i] = s/n
            elseif i == n
                out[i] = s/n
            else
                out[i] = NaN
            end
        end
        return out
    end

    function positions(s::MomentumTS, df::DataFrame, p::NamedTuple)
        price = Float64.(df[!, s.col_price])
        fast = ma(price, p.fast)
        slow = ma(price, p.slow)
        sig = map((a,b)->(isnan(a) || isnan(b)) ? 0.0 : (a>b ? 1.0 : (s.allow_short ? -1.0 : 0.0)), fast, slow)
        # CORREÇÃO: Atraso adaptativo - 1 dia para períodos curtos, 2 para longos
        lag = length(sig) < 100 ? 1 : 2
        pos = length(sig) > lag ? vcat(zeros(lag), sig[1:end-lag]) : zeros(length(sig))
        return pos
    end

    # --- Pairs Trading ---
    mutable struct PairsCoint <: Strategy
        col_p1::Symbol
        col_p2::Symbol  
        hedgeβ::Float64
        use_logs::Bool
        dynamic_window::Bool
    end
    PairsCoint(;col_p1=:Close1, col_p2=:Close2, hedgeβ=NaN, use_logs=true, dynamic_window=true) = 
        PairsCoint(col_p1, col_p2, hedgeβ, use_logs, dynamic_window)

    function param_grid(::PairsCoint)
        return [(z_enter=2.0, z_exit=0.5), (z_enter=1.5, z_exit=0.5), (z_enter=2.5, z_exit=1.0)]
    end

    function fit!(s::PairsCoint, df_train::DataFrame)
        p1 = Float64.(df_train[!, s.col_p1])
        p2 = Float64.(df_train[!, s.col_p2])
        if s.use_logs
            y = log.(p1); x = log.(p2)
        else
            y = p1; x = p2
        end
        β = var(x) == 0 ? 1.0 : cov(x,y)/var(x)
        s.hedgeβ = β
        return nothing
    end

    function movmean(x::AbstractVector, n::Int)
        n <= 1 && return copy(x)
        out = similar(collect(x), Float64)
        s = 0.0
        for i in eachindex(x)
            s += x[i]
            if i > n
                s -= x[i-n]
                out[i] = s/n
            elseif i == n
                out[i] = s/n
            else
                out[i] = NaN
            end
        end
        return out
    end

    function movstd(x::AbstractVector, n::Int)
        m = movmean(x, n)
        out = similar(collect(x), Float64)
        for i in eachindex(x)
            if i < n || isnan(m[i])
                out[i] = NaN
            else
                window = view(x, i-n+1:i)
                out[i] = std(window)
            end
        end
        return out
    end

    function positions(s::PairsCoint, df::DataFrame, p::NamedTuple)
        p1 = Float64.(df[!, s.col_p1])
        p2 = Float64.(df[!, s.col_p2])
        y = s.use_logs ? log.(p1) : p1
        x = s.use_logs ? log.(p2) : p2
        β = isfinite(s.hedgeβ) ? s.hedgeβ : 1.0
        spread = y .- β .* x
        
        window_size = s.dynamic_window ? max(20, min(80, div(length(spread), 3))) : 30
        
        μ = movmean(spread, window_size)
        σ = movstd(spread, window_size)
        z = (spread .- μ) ./ σ
        sig = Vector{Float64}(undef, length(z))
        prev = 0.0
        for i in eachindex(z)
            zi = z[i]
            if isnan(zi)
                sig[i] = prev
            elseif zi > p.z_enter
                sig[i] = -1.0
            elseif zi < -p.z_enter
                sig[i] = 1.0
            elseif abs(zi) < p.z_exit
                sig[i] = 0.0
            else
                sig[i] = prev
            end
            prev = sig[i]
        end
        # CORREÇÃO: Atraso adaptativo - 1 dia para períodos curtos, 2 para longos
        lag = length(sig) < 100 ? 1 : 2
        pos = length(sig) > lag ? vcat(zeros(lag), sig[1:end-lag]) : zeros(length(sig))
        return pos
    end

    function pairs_gross_returns(df::DataFrame, s::PairsCoint, pos::AbstractVector)
        p1 = Float64.(df[!, s.col_p1])
        p2 = Float64.(df[!, s.col_p2])
        r1 = [0.0; diff(log.(p1))]
        r2 = [0.0; diff(log.(p2))]
        β  = isfinite(s.hedgeβ) ? s.hedgeβ : 1.0
        # CORREÇÃO: Hedge ratio simétrico para neutralizar exposição
        total_weight = 1.0 + abs(β)
        w1 = pos ./ total_weight
        w2 = -β .* pos ./ total_weight
        return w1 .* r1 .+ w2 .* r2
    end

    # --- Sazonalidade ---
    mutable struct Seasonality <: Strategy
        col_price::Symbol
        selected_months::Vector{Int}
        k::Int
    end
    Seasonality(;col_price=:Close, selected_months=Int[], k=3) = 
        Seasonality(col_price, selected_months, k)

    function param_grid(s::Seasonality)
        return [(k=4,), (k=6,), (k=8,)]  # Mais meses para aumentar overlap com períodos de 12 meses
    end

    function fit!(s::Seasonality, df_train::DataFrame)
        # Armazenar dados de treino para evitar look-ahead bias no positions()
        price = Float64.(df_train[!, s.col_price])
        dates = Date.(df_train[!, :Date])
        r = [0.0; diff(log.(price))]
        df = DataFrame(Date=dates, r=r)
        df.Month = Dates.month.(df.Date)
        g = combine(groupby(df, :Month), :r => mean => :avg)
        sort!(g, :avg, rev=true)
        
        # Armazenar ranking de meses para uso no positions()
        s.selected_months = collect(g.Month)  # Lista ordenada por performance
        return nothing
    end

    function positions(s::Seasonality, df::DataFrame, p::NamedTuple)
        # CORREÇÃO: Usar ranking de meses do treino (sem look-ahead bias)
        dates = Date.(df[!, :Date])
        
        # Usar top k meses baseado no ranking do treino
        top_k_months = s.selected_months[1:min(p.k, length(s.selected_months))]
        sel = Set(top_k_months)
        
        sig = [Dates.month(d) in sel ? 1.0 : 0.0 for d in dates]
        # CORREÇÃO: Atraso adaptativo - 1 dia para períodos curtos, 2 para longos
        lag = length(sig) < 100 ? 1 : 2
        pos = length(sig) > lag ? vcat(zeros(lag), sig[1:end-lag]) : zeros(length(sig))
        return pos
    end

end