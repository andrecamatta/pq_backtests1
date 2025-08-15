module Metrics
    using Statistics, StatsBase, Distributions, Dates, Random
    export sharpe, max_drawdown, cagr, mar, psr, dsr, 
           weighted_mean_metric, bootstrap_ci, compound_sharpe, newey_west_sharpe

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

    # ========== MÉTRICAS ACADÊMICAS ROBUSTAS ==========
    
    """
    Calcula média ponderada de métricas por split, corrigindo o viés da concatenação.
    Pesos baseados no número de observações por split.
    """
    function weighted_mean_metric(metric_func, split_returns::Vector{Vector{Float64}}, 
                                 split_weights::Vector{Int}; kwargs...)
        if isempty(split_returns)
            return NaN, NaN, NaN  # média, std_err, p_value
        end
        
        # Calcular métrica para cada split
        split_metrics = Float64[]
        weights = Float64[]
        
        for (i, rets) in enumerate(split_returns)
            if !isempty(rets) && sum(abs.(rets)) > 1e-6  # Validar trades suficientes
                metric_val = metric_func(rets; kwargs...)
                if isfinite(metric_val)
                    push!(split_metrics, metric_val)
                    push!(weights, Float64(split_weights[i]))
                end
            end
        end
        
        if isempty(split_metrics)
            return NaN, NaN, NaN
        end
        
        # Média ponderada
        w_norm = weights ./ sum(weights)
        weighted_avg = sum(split_metrics .* w_norm)
        
        # Erro padrão ponderado (Cochran 1977)
        if length(split_metrics) > 1
            weighted_var = sum(w_norm .* (split_metrics .- weighted_avg).^2)
            # Correção para pesos desiguais
            eff_n = sum(weights)^2 / sum(weights.^2)
            std_err = sqrt(weighted_var / eff_n)
            
            # Teste t para H0: métrica = 0
            if std_err > 1e-10
                t_stat = weighted_avg / std_err
                df = length(split_metrics) - 1
                p_value = 2 * (1 - cdf(TDist(df), abs(t_stat)))
            else
                p_value = weighted_avg == 0.0 ? 1.0 : 0.0
            end
        else
            std_err = NaN
            p_value = NaN
        end
        
        return weighted_avg, std_err, p_value
    end
    
    """
    Bootstrap confidence intervals para métricas.
    """
    function bootstrap_ci(metric_func, returns::AbstractVector; 
                         confidence::Float64=0.95, n_bootstrap::Int=1000, kwargs...)
        if length(returns) < 10
            return NaN, NaN
        end
        
        Random.seed!(42)  # Reproduzibilidade
        bootstrap_values = Float64[]
        
        for _ in 1:n_bootstrap
            boot_sample = sample(returns, length(returns), replace=true)
            boot_metric = metric_func(boot_sample; kwargs...)
            if isfinite(boot_metric)
                push!(bootstrap_values, boot_metric)
            end
        end
        
        if isempty(bootstrap_values)
            return NaN, NaN
        end
        
        α = 1 - confidence
        lower_percentile = 100 * (α/2)
        upper_percentile = 100 * (1 - α/2)
        
        ci_lower = percentile(bootstrap_values, lower_percentile)
        ci_upper = percentile(bootstrap_values, upper_percentile)
        
        return ci_lower, ci_upper
    end
    
    """
    Compound Sharpe Ratio: média geométrica de Sharpe por split.
    Mais robusta a outliers que média aritmética.
    """
    function compound_sharpe(split_returns::Vector{Vector{Float64}}; freq::Int=252, kwargs...)
        if isempty(split_returns)
            return NaN
        end
        
        sharpe_values = Float64[]
        for rets in split_returns
            if !isempty(rets) && sum(abs.(rets)) > 1e-6
                sr = sharpe(rets; freq=freq, kwargs...)
                if isfinite(sr) && sr > 0  # Geometric mean requer valores positivos
                    push!(sharpe_values, sr)
                end
            end
        end
        
        if isempty(sharpe_values)
            return NaN
        end
        
        # Média geométrica com shift para lidar com valores negativos
        min_sr = minimum(sharpe_values)
        if min_sr <= 0
            shift = abs(min_sr) + 0.1
            shifted_values = sharpe_values .+ shift
            geom_mean_shifted = exp(mean(log.(shifted_values)))
            return geom_mean_shifted - shift
        else
            return exp(mean(log.(sharpe_values)))
        end
    end
    
    """
    Newey-West Sharpe Ratio com correção para autocorrelação.
    """
    function newey_west_sharpe(returns::AbstractVector; freq::Int=252, lags::Int=0, kwargs...)
        r = collect(skipmissing(returns))
        n = length(r)
        
        if n < 10
            return NaN, NaN
        end
        
        μ = mean(r)
        
        # Auto-select optimal lags se não especificado
        if lags == 0
            lags = min(floor(Int, 4*(n/100)^(2/9)), n÷4)  # Newey-West (1987)
        end
        
        # Covariância com correção HAC (Heteroscedasticity and Autocorrelation Consistent)
        γ0 = var(r)  # lag 0
        nw_var = γ0
        
        for j in 1:lags
            if j < n
                # Autocovariance lag j
                γj = sum((r[1:end-j] .- μ) .* (r[j+1:end] .- μ)) / n
                # Bartlett kernel weight
                weight = 1 - j/(lags + 1)
                nw_var += 2 * weight * γj
            end
        end
        
        if nw_var <= 0
            return NaN, NaN
        end
        
        nw_sharpe = (μ / sqrt(nw_var)) * sqrt(freq)
        
        # Standard error for Sharpe ratio (Lo 2002)
        sharpe_se = sqrt((1 + 0.5 * nw_sharpe^2) / n) * sqrt(freq)
        
        return nw_sharpe, sharpe_se
    end
    
end