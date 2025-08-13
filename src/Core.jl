module Core

    using Dates, DataFrames
    using ..Config, ..Metrics, ..Costs, ..Split, ..Strategies

    export RunResult, run_wfo, summarize_result

    struct RunResult
        strategy_name::String
        config_name::String
        dates::Vector{Date}
        oos_returns::Vector{Float64}
        oos_equity::Vector{Float64}
        best_params_history::Vector{NamedTuple}
        split_dates::Vector{Vector{Date}}
        split_returns::Vector{Vector{Float64}}
    end

    function single_asset_gross_returns(price::AbstractVector)
        return [0.0; diff(log.(Float64.(price)))]
    end

    function grid_search_params(strategy, df_train::DataFrame, cfg; price_col::Symbol=:Close)
        # CORREÇÃO: Validar tamanho mínimo de amostra para otimização robusta
        min_samples = 60  # Mínimo 60 observações para Sharpe ratio confiável
        if nrow(df_train) < min_samples
            @warn "Amostra de treino muito pequena ($(nrow(df_train)) < $min_samples). Usando primeiro parâmetro."
            return first(param_grid(strategy))
        end
        
        best_p, best_score = nothing, -Inf
        for p in param_grid(strategy)
            pos = positions(strategy, df_train, p)
            
            if typeof(strategy) == PairsCoint
                gross = pairs_gross_returns(df_train, strategy, pos)
            else
                gross = single_asset_gross_returns(df_train[!, price_col]) .* pos
            end
            
            net = net_returns(gross, pos; commission=cfg.commission, slippage=cfg.slippage)
            # Validar se há trades suficientes
            total_trades = sum(abs.(diff(pos)))
            if total_trades < 1e-6  # Sem trades
                @warn "Parâmetro $(p) não gerou trades no período de treino"
                continue
            end
            
            sc = sharpe(net; freq=cfg.trading_days)
            if sc > best_score
                best_score, best_p = sc, p
            end
        end
        return best_p === nothing ? first(param_grid(strategy)) : best_p
    end

    function run_wfo(strategy, df::DataFrame, cfg, config_name::String; price_col::Symbol=:Close)
        dates = Date.(df[!, :Date])
        splits = walk_forward_splits(dates;
            train_months=cfg.train_months, test_months=cfg.test_months,
            embargo_days=cfg.embargo_days, label_horizon_days=cfg.label_horizon_days)

        oos_rets = Float64[]
        best_hist = NamedTuple[]
        split_dates = Vector{Date}[]
        split_returns = Vector{Float64}[]

        for sp in splits
            df_train = df[sp.train_idx, :]
            df_test  = df[sp.test_idx, :]
            
            fit!(strategy, df_train)
            best_p = grid_search_params(strategy, df_train, cfg; price_col)
            push!(best_hist, best_p)
            
            pos = positions(strategy, df_test, best_p)
            
            if typeof(strategy) == PairsCoint
                gross = pairs_gross_returns(df_test, strategy, pos)
            else
                gross = single_asset_gross_returns(df_test[!, price_col]) .* pos
            end
            
            net = net_returns(gross, pos; commission=cfg.commission, slippage=cfg.slippage)
            
            # VALIDAÇÃO: Alertar se não há trades no período de teste
            total_trades = sum(abs.(diff(pos)))
            if total_trades < 1e-6
                @warn "Split sem trades: parâmetro $(best_p) não gerou posições no período de teste"
            end
            
            append!(oos_rets, net)
            
            push!(split_dates, dates[sp.test_idx])
            push!(split_returns, net)
        end

        eq = accumulate((x,y)->x*(1+y), oos_rets; init=1.0)[2:end]
        oos_dates = vcat(split_dates...)  # Concatena datas reais dos períodos OOS
        return RunResult(string(typeof(strategy)), config_name, oos_dates, oos_rets, eq, best_hist, split_dates, split_returns)
    end

    function summarize_result(result::RunResult; freq::Int=252, n_trials::Int=1)
        r = result.oos_returns
        eq = result.oos_equity
        S  = sharpe(r; freq=freq)
        MDD = max_drawdown(eq)
        CAGR = cagr(eq; freq=freq)
        MAR = mar(eq, r; freq=freq)
        PSR = psr(r; freq=freq, sr_benchmark=0.0)
        DSR = dsr(r; freq=freq, n_trials=n_trials)
        
        # Calcular métricas por split
        split_metrics = []
        for (i, (dates, rets)) in enumerate(zip(result.split_dates, result.split_returns))
            if !isempty(rets)
                eq_split = accumulate((x,y)->x*(1+y), rets; init=1.0)[2:end]
                push!(split_metrics, (
                    split=i,
                    test_dates=(first(dates), last(dates)),
                    sharpe=sharpe(rets; freq=freq),
                    cagr=cagr(eq_split; freq=freq),
                    max_dd=max_drawdown(eq_split),
                    mar=mar(eq_split, rets; freq=freq),
                    psr=psr(rets; freq=freq, sr_benchmark=0.0),
                    dsr=dsr(rets; freq=freq, n_trials=n_trials),
                    n_days=length(rets)
                ))
            end
        end
        
        return (
            strategy=result.strategy_name,
            config=result.config_name,
            sharpe=S,
            cagr=CAGR,
            max_dd=MDD,
            mar=MAR,
            psr=PSR,
            dsr=DSR,
            n_oos=length(r),
            n_splits=length(result.best_params_history),
            split_metrics=split_metrics
        )
    end
end