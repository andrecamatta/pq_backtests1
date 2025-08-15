module Strategies

    using DataFrames, Dates, Statistics, StatsBase
    export Strategy, MomentumTS, PairsCoint, Seasonality, param_grid, fit!, positions, pairs_gross_returns, month_names

    abstract type Strategy end

    # --- Momentum ---
    struct MomentumTS <: Strategy
        col_price::Symbol
        allow_short::Bool
    end
    MomentumTS(;col_price=:Close, allow_short=true) = MomentumTS(col_price, allow_short)

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
        # Armazenar meses selecionados para cada parâmetro testado
        last_top_months::Vector{Int}
        last_bottom_months::Vector{Int}
    end
    Seasonality(;col_price=:Close, selected_months=Int[], k=3) = 
        Seasonality(col_price, selected_months, k, Int[], Int[])

    function param_grid(s::Seasonality)
        # CORREÇÃO: Param grid limitado para evitar look-ahead bias
        # k máximo = 6 para manter seleção mutuamente exclusiva (6 long + 6 short = 12 meses)
        return [(k=2,), (k=4,), (k=6,)]  # Progressão conservativa sem bias
    end

    function fit!(s::Seasonality, df_train::DataFrame)
        # CORREÇÃO LOOK-AHEAD BIAS: Não calcular ranking antecipadamente
        # O ranking deve ser calculado iterativamente na função positions()
        # usando apenas dados históricos até cada data específica
        
        # Apenas inicializar se necessário - sem look-ahead
        s.selected_months = Int[]  # Vazio - será calculado dinamicamente
        return nothing
    end

    function positions(s::Seasonality, df::DataFrame, p::NamedTuple)
        # CORREÇÃO DEFINITIVA DO LOOK-AHEAD BIAS: Ranking móvel baseado apenas em dados passados
        dates = Date.(df[!, :Date])
        prices = Float64.(df[!, s.col_price])
        sig = zeros(length(dates))
        
        # Parâmetros para ranking móvel (ajustados para compatibilidade)
        min_lookback_days = 180  # Reduzido para 6 meses - balance entre robustez e viabilidade
        k_effective = min(p.k, 6)  # Máximo 6 meses para evitar sobreposição
        
        for i in eachindex(dates)
            current_date = dates[i]
            current_month = Dates.month(current_date)
            
            # Usar apenas dados históricos até ONTEM (i-1)
            if i > min_lookback_days
                # Slice histórico: do início até ontem (SEM look-ahead)
                hist_prices = prices[1:i-1]
                
                # Calcular retornos históricos (diff remove primeira observação)
                hist_log_returns = diff(log.(hist_prices))
                # Alinhar datas com retornos: retornos começam da segunda data
                hist_dates_aligned = dates[2:i-1]  # Corresponde aos retornos
                
                # DEBUG: Verificar alinhamento
                if length(hist_dates_aligned) != length(hist_log_returns)
                    @warn "Desalinhamento detectado: dates=$(length(hist_dates_aligned)), returns=$(length(hist_log_returns))"
                end
                
                # Criar DataFrame histórico com dimensões corretas
                if !isempty(hist_log_returns) && !isempty(hist_dates_aligned)
                    hist_df = DataFrame(Date=hist_dates_aligned, Returns=hist_log_returns)
                    hist_df.Month = Dates.month.(hist_df.Date)
                
                    # Calcular ranking de meses usando APENAS dados passados
                    monthly_performance = combine(groupby(hist_df, :Month), :Returns => mean => :avg_return)
                    sort!(monthly_performance, :avg_return, rev=true)
                    
                    # Logging básico
                    n_months = nrow(monthly_performance)
                    
                    # Validação menos restritiva: pelo menos k_effective meses únicos
                    min_required_months = min(k_effective + 2, 8)  # Mais flexível
                    
                    if n_months >= min_required_months
                        # Ajustar k_effective se não há meses suficientes
                        k_actual = min(k_effective, n_months ÷ 2)
                        
                        if k_actual > 0
                            top_months = Set(monthly_performance[1:k_actual, :Month])
                            bottom_months = Set(monthly_performance[end-k_actual+1:end, :Month])
                            
                            # Armazenar meses selecionados para uso na saída (apenas na primeira execução válida)
                            if isempty(s.last_top_months)
                                s.last_top_months = sort(collect(top_months))
                                s.last_bottom_months = sort(collect(bottom_months))
                            end
                            
                            # Determinar posição para o mês atual
                            if current_month in top_months
                                sig[i] = 1.0   # Long nos meses historicamente bons
                            elseif current_month in bottom_months
                                sig[i] = -1.0  # Short nos meses historicamente ruins
                            else
                                sig[i] = 0.0   # Neutro nos demais
                            end
                        else
                            sig[i] = 0.0  # k_actual = 0
                        end
                    else
                        sig[i] = 0.0  # Não há meses suficientes
                    end
                else
                    sig[i] = 0.0  # DataFrame vazio
                end
            else
                sig[i] = 0.0  # Período inicial sem dados suficientes
            end
        end
        
        # Aplicar lag adaptativo para evitar execução no mesmo dia
        lag = length(sig) < 100 ? 1 : 2
        pos = length(sig) > lag ? vcat(zeros(lag), sig[1:end-lag]) : zeros(length(sig))
        
        return pos
    end
    
    # Helper function para converter números de meses para nomes abreviados
    function month_names(months::Vector{Int})
        month_abbrev = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", 
                       "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
        return [month_abbrev[m] for m in months if 1 <= m <= 12]
    end

end