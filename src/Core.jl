module Core

    using Dates, DataFrames, Statistics
    using ..Config, ..Metrics, ..Costs, ..Split, ..Strategies

    export RunResult, run_wfo, summarize_result

    struct RunResult
        strategy_name::String
        config_name::String
        dates::Vector{Date}
        oos_returns::Vector{Float64}
        oos_equity::Vector{Float64}
        bh_returns::Vector{Float64}  # Buy-and-hold returns
        best_params_history::Vector{NamedTuple}
        split_dates::Vector{Vector{Date}}
        split_returns::Vector{Vector{Float64}}
        split_bh_returns::Vector{Vector{Float64}}
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
        bh_rets = Float64[]  # Buy-and-hold returns
        best_hist = NamedTuple[]
        split_dates = Vector{Date}[]
        split_returns = Vector{Float64}[]
        split_bh_returns = Vector{Float64}[]

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
            
            # Calcular buy-and-hold returns 
            if typeof(strategy) != PairsCoint
                # Estratégias single-asset: usar buy-and-hold do ativo
                bh_gross = single_asset_gross_returns(df_test[!, price_col])
                bh_net = net_returns(bh_gross, ones(length(bh_gross)); commission=0.0, slippage=0.0)  # BH sem custos
                append!(bh_rets, bh_net)
                push!(split_bh_returns, bh_net)
            else
                # Para pairs trading: benchmark 50% cada ativo
                bh_gross1 = single_asset_gross_returns(df_test[!, :Close1])
                bh_gross2 = single_asset_gross_returns(df_test[!, :Close2])
                bh_gross_50_50 = 0.5 * bh_gross1 + 0.5 * bh_gross2
                bh_net = net_returns(bh_gross_50_50, ones(length(bh_gross_50_50)); commission=0.0, slippage=0.0)
                append!(bh_rets, bh_net)
                push!(split_bh_returns, bh_net)
            end
            
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
        return RunResult(string(typeof(strategy)), config_name, oos_dates, oos_rets, eq, bh_rets, best_hist, split_dates, split_returns, split_bh_returns)
    end

    function summarize_result(result::RunResult; freq::Int=252, n_trials::Int=1)
        # MÉTRICAS ACADÊMICAS ROBUSTAS: Usar splits ao invés de concatenação
        split_returns = result.split_returns
        split_weights = [length(rets) for rets in split_returns]  # Peso = número observações
        
        # Calcular métricas robustas por split
        S, S_se, S_pval = weighted_mean_metric(sharpe, split_returns, split_weights; freq=freq)
        compound_S = compound_sharpe(split_returns; freq=freq)
        
        # Dados básicos para benchmark calculation
        r = result.oos_returns
        bh_r = result.bh_returns
        
        # DSR usando média ponderada por split
        DSR, DSR_se, DSR_pval = weighted_mean_metric(dsr, split_returns, split_weights; freq=freq, n_trials=n_trials)
        
        # Newey-West Sharpe para correção de autocorrelação
        nw_sharpe, nw_se = newey_west_sharpe(r; freq=freq)
        
        # Calcular métricas do benchmark com abordagem robusta
        has_benchmark = !all(x -> abs(x) < 1e-10, bh_r)
        if has_benchmark
            # Benchmark metrics usando splits (abordagem robusta)
            split_bh_returns = result.split_bh_returns
            S_BH, S_BH_se, S_BH_pval = weighted_mean_metric(sharpe, split_bh_returns, split_weights; freq=freq)
            
            # ADJUSTED INFORMATION RATIO POR SPLIT (mais robusto)
            split_info_ratios = Float64[]
            split_alphas = Float64[]
            valid_split_indices = Int[]  # Track which splits are valid
            
            for (i, (rets, bh_rets)) in enumerate(zip(split_returns, split_bh_returns))
                if !isempty(rets) && !isempty(bh_rets) && sum(abs.(rets)) > 1e-6
                    excess_r = rets .- bh_rets
                    if !isempty(excess_r)
                        alpha_split = mean(excess_r) * freq
                        tracking_error_split = std(excess_r) * sqrt(freq)
                        
                        if tracking_error_split > 1e-10
                            ir_split = alpha_split / tracking_error_split
                            push!(split_info_ratios, ir_split)
                            push!(split_alphas, alpha_split)
                            push!(valid_split_indices, i)  # Track valid split index
                        end
                    end
                end
            end
            
            # Média ponderada das métricas de alpha usando splits válidos
            if !isempty(split_info_ratios)
                valid_weights = [split_weights[i] for i in valid_split_indices]  # Fix: get correct weights
                w_norm = valid_weights ./ sum(valid_weights)
                info_ratio = sum(split_info_ratios .* w_norm)
                alpha = sum(split_alphas .* w_norm)
                
                # Tracking error agregado - CONSISTENTE com alpha calculation
                # Usar splits válidos ao invés de dados concatenados
                all_excess_returns = Float64[]
                all_weights_expanded = Float64[]
                
                for (i, idx) in enumerate(valid_split_indices)
                    rets = split_returns[idx]
                    bh_rets = split_bh_returns[idx]
                    excess_r_split = rets .- bh_rets
                    weight_split = valid_weights[i]
                    
                    append!(all_excess_returns, excess_r_split)
                    append!(all_weights_expanded, fill(weight_split, length(excess_r_split)))
                end
                
                if !isempty(all_excess_returns)
                    # Weighted tracking error para consistência com weighted alpha
                    w_te = all_weights_expanded ./ sum(all_weights_expanded)
                    weighted_mean_excess = sum(all_excess_returns .* w_te)
                    weighted_var_excess = sum(w_te .* (all_excess_returns .- weighted_mean_excess).^2)
                    tracking_error = sqrt(weighted_var_excess) * sqrt(freq)
                    
                    # VALIDAÇÃO MATEMÁTICA: IR deve ser consistente
                    expected_ir = tracking_error > 1e-10 ? alpha / tracking_error : 0.0
                    ir_diff = abs(info_ratio - expected_ir)
                    
                    if ir_diff > 1e-4  # Tolerância para diferenças numéricas
                        @warn "Information Ratio inconsistency detected: IR=$(info_ratio), Expected=$(expected_ir), Diff=$(ir_diff)"
                        # Use the mathematically consistent version
                        info_ratio = expected_ir
                    end
                else
                    tracking_error = NaN
                end
            else
                info_ratio = alpha = tracking_error = NaN
            end
            
            # Deflated Information Ratio usando DSR robusto
            dir = (isfinite(info_ratio) && info_ratio != 0.0 && isfinite(DSR) && DSR > 0.0) ? 
                  info_ratio * sqrt(DSR) : 0.0
        else
            S_BH = S_BH_se = S_BH_pval = alpha = tracking_error = info_ratio = dir = NaN
        end
        
        # Calcular métricas por split
        split_metrics = []
        for (i, (dates, rets, bh_rets_split)) in enumerate(zip(result.split_dates, result.split_returns, result.split_bh_returns))
            if !isempty(rets)
                eq_split = accumulate((x,y)->x*(1+y), rets; init=1.0)[2:end]
                
                # Detectar se houve trades suficientes
                # Aproximação: se todos os retornos são zero ou quase zero, não houve trades
                total_abs_returns = sum(abs.(rets))
                has_sufficient_trades = total_abs_returns > 1e-6
                
                if has_sufficient_trades
                    # Calcular benchmark para este split
                    has_bh_split = !all(x -> abs(x) < 1e-10, bh_rets_split)
                    dsr_split = dsr(rets; freq=freq, n_trials=n_trials)
                    
                    if has_bh_split
                        bh_eq_split = accumulate((x,y)->x*(1+y), bh_rets_split; init=1.0)[2:end]
                        sharpe_bh_split = sharpe(bh_rets_split; freq=freq)
                        alpha_split = mean(rets .- bh_rets_split) * freq
                        tracking_error_split = std(rets .- bh_rets_split) * sqrt(freq)
                        info_ratio_split = tracking_error_split > 1e-10 ? alpha_split / tracking_error_split : 0.0
                        
                        # Calcular DIR para este split
                        dir_split = (info_ratio_split != 0.0 && dsr_split > 0.0 && !isnan(dsr_split)) ? info_ratio_split * sqrt(dsr_split) : 0.0
                    else
                        sharpe_bh_split = alpha_split = info_ratio_split = dir_split = NaN
                    end
                    
                    push!(split_metrics, (
                        split=i,
                        test_dates=(first(dates), last(dates)),
                        sharpe=sharpe(rets; freq=freq),
                        sharpe_bh=sharpe_bh_split,
                        alpha=alpha_split,
                        dsr=dsr_split,
                        dir=dir_split,
                        n_days=length(rets),
                        has_trades=true
                    ))
                else
                    # Sem trades suficientes - retornar NaN para todas as métricas
                    push!(split_metrics, (
                        split=i,
                        test_dates=(first(dates), last(dates)),
                        sharpe=NaN,
                        sharpe_bh=NaN,
                        alpha=NaN,
                        dsr=NaN,
                        dir=NaN,
                        n_days=length(rets),
                        has_trades=false
                    ))
                end
            end
        end
        
        return (
            strategy=result.strategy_name,
            config=result.config_name,
            # Métricas robustas com erro padrão e p-value
            sharpe=S,
            sharpe_se=S_se,
            sharpe_pval=S_pval,
            sharpe_compound=compound_S,
            sharpe_nw=nw_sharpe,
            sharpe_nw_se=nw_se,
            # Benchmark
            sharpe_bh=S_BH,
            sharpe_bh_se=S_BH_se,
            sharpe_bh_pval=S_BH_pval,
            # Alpha metrics
            alpha=alpha,
            info_ratio=info_ratio,
            tracking_error=tracking_error,
            dir=dir,
            # Outras métricas (removidas métricas legacy não utilizadas)
            dsr=DSR,
            dsr_se=DSR_se,
            dsr_pval=DSR_pval,
            # Meta info
            has_benchmark=has_benchmark,
            n_oos=length(result.oos_returns),
            n_splits=length(result.best_params_history),
            split_metrics=split_metrics
        )
    end
end