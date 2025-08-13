module Metrics
    using Statistics, StatsBase, Distributions, Dates
    export sharpe, max_drawdown, cagr, mar, psr, dsr

    function sharpe(returns::AbstractVector; freq::Int=252, rf_per_period::Float64=0.0)
        r = collect(skipmissing(returns .- rf_per_period))
        length(r) == 0 && return 0.0
        μ = mean(r)
        σ = std(r)
        (σ == 0.0 || isnan(σ)) && return 0.0
        return (μ/σ) * sqrt(freq)
    end

    function max_drawdown(equity::AbstractVector)
        peak = -Inf
        mdd = 0.0
        for v in equity
            peak = max(peak, v)
            dd = v/peak - 1
            mdd = min(mdd, dd)
        end
        return mdd
    end

    function cagr(equity::AbstractVector; freq::Int=252)
        n = length(equity)
        n <= 1 && return 0.0
        # CORREÇÃO: Usar o número real de períodos, não assumir freq
        years = (n-1) / freq
        return (equity[end]/equity[1])^(1/years) - 1
    end

    function mar(equity::AbstractVector, returns::AbstractVector; freq::Int=252)
        c = cagr(equity; freq)
        mdd = abs(max_drawdown(equity))
        # CORREÇÃO: Retornar valor finito quando MDD ≈ 0 (sem drawdown significativo)
        mdd < 1e-8 ? (c < 0 ? -1000.0 : 1000.0) : c/mdd
    end

    function psr(returns::AbstractVector; freq::Int=252, sr_benchmark::Float64=0.0)
        r = collect(skipmissing(returns))
        T = length(r)
        T < 10 && return 0.0
        μ = mean(r)
        σ = std(r)
        # CORREÇÃO: Tratar casos de σ=0 (sem variabilidade)
        if σ == 0.0 || isnan(σ)
            return μ >= sr_benchmark ? 1.0 : 0.0
        end
        
        sk = skewness(r)
        kt = kurtosis(r)
        γ3 = sk
        γ4 = kt + 3
        SR = (μ/σ) * sqrt(freq)
        
        # CORREÇÃO: Validar expressão sob raiz quadrada
        expr = 1 - γ3*SR + 0.25*(γ4-1)*SR^2
        if expr <= 0.0 || isnan(expr)
            # Usar fórmula simplificada quando há problemas com momentos superiores
            return cdf(Normal(), SR * sqrt(T))
        end
        
        denom = sqrt(expr)
        z = (SR - sr_benchmark) * sqrt(T) / denom
        return cdf(Normal(), z)
    end

    function dsr(returns::AbstractVector; freq::Int=252, n_trials::Int=1)
        r = collect(skipmissing(returns))
        T = length(r)
        T < 10 && return 0.0
        μ = mean(r)
        σ = std(r)
        # CORREÇÃO: Tratar casos de σ=0
        if σ == 0.0 || isnan(σ)
            return μ >= 0.0 ? 1.0 : 0.0
        end
        
        sk = skewness(r)
        kt = kurtosis(r)
        γ3 = sk
        γ4 = kt + 3
        SR = (μ/σ) * sqrt(freq)
        
        # CORREÇÃO: Validar expressão sob raiz quadrada
        expr = (1 - γ3*SR + 0.25*(γ4-1)*SR^2) / T
        if expr <= 0.0 || isnan(expr)
            # Usar fórmula simplificada
            σ_SR = 1.0 / sqrt(T)
        else
            σ_SR = sqrt(expr)
        end
        
        n_trials <= 1 && return cdf(Normal(), SR/σ_SR)
        SR0 = quantile(Normal(), 1 - 1/Float64(n_trials)) * σ_SR
        z = (SR - SR0) / σ_SR
        return cdf(Normal(), z)
    end
end