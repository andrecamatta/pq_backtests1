#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

# Inclui o módulo principal que agora contém toda a lógica
include("src/PQBacktester.jl")
using .PQBacktester
using Dates, DataFrames

function run_all_strategies()
    println("🚀 Iniciando Backtest Quantitativo com Configurações Otimizadas")
    println("="^75)
    
    # --- Carregamento de Dados ---
    println("📊 Carregando dados REAIS...")
    ibov_data = load_real_data("^BVSP"; start_date=Date(2015,1,1), end_date=Date(2024,12,31))
    suzb_data = load_real_data("SUZB3.SA"; start_date=Date(2015,1,1), end_date=Date(2024,12,31))
    klbn_data = load_real_data("KLBN11.SA"; start_date=Date(2015,1,1), end_date=Date(2024,12,31))
    
    if isempty(ibov_data) || isempty(suzb_data) || isempty(klbn_data)
        println("❌ Falha no carregamento de dados. Abortando.")
        return
    end
    
    df_mom = prepare_single_asset(copy(ibov_data))
    df_pair = prepare_pair(copy(suzb_data), copy(klbn_data))
    
    println("✅ Dados carregados: IBOV($(nrow(df_mom))), SUZB3/KLBN11 Pair($(nrow(df_pair)))")
    
    # --- Execução das Estratégias ---

    # 1. Estratégia Momentum
    println("\n" * "="^50)
    println("1️⃣  Executando Estratégia: MOMENTUM")
    println("="^50)
    cfg_mom = MomentumConfig()
    strategy_mom = MomentumTS()
    res_mom = run_wfo(strategy_mom, copy(df_mom), cfg_mom, "Otimizada"; price_col=:Close)
    summary_mom = summarize_result(res_mom; n_trials=3)
    println("📈 Resultados por Split:")
    for m in summary_mom.split_metrics
        println("   Split $(m.split) (Período de Teste: $(m.test_dates[1]) até $(m.test_dates[2])):")
        println("      MAR: $(round(m.mar, digits=6)) | PSR: $(round(m.psr, digits=6)) | DSR: $(round(m.dsr, digits=6)) | Dias: $(m.n_days)")
        println("      📊 Parâmetro: Compra quando Média $(res_mom.best_params_history[m.split].fast)d > Média $(res_mom.best_params_history[m.split].slow)d")
    end
    println("   📊 Resultado Geral (média de todos os splits):")
    println("      MAR: $(round(summary_mom.mar, digits=6)) | PSR: $(round(summary_mom.psr, digits=6)) | DSR: $(round(summary_mom.dsr, digits=6))")

    # 2. Estratégia Pairs Trading
    println("\n" * "="^50)
    println("2️⃣  Executando Estratégia: PAIRS TRADING (Janela Dinâmica)")
    println("="^50)
    cfg_pair = PairsConfig()
    strategy_pair = PairsCoint(dynamic_window=true)
    res_pair = run_wfo(strategy_pair, copy(df_pair), cfg_pair, "Otimizada")
    summary_pair = summarize_result(res_pair; n_trials=3)
    println("📊 Resultados por Split:")
    for m in summary_pair.split_metrics
        println("   Split $(m.split) (Período de Teste: $(m.test_dates[1]) até $(m.test_dates[2])):")
        println("      MAR: $(round(m.mar, digits=6)) | PSR: $(round(m.psr, digits=6)) | DSR: $(round(m.dsr, digits=6)) | Dias: $(m.n_days)")
        println("      📊 Parâmetro: Compra quando z-score > $(res_pair.best_params_history[m.split].z_enter) ou z-score < -$(res_pair.best_params_history[m.split].z_enter), Venda quando |z-score| < $(res_pair.best_params_history[m.split].z_exit)")
    end
    println("   📊 Resultado Geral (média de todos os splits):")
    println("      MAR: $(round(summary_pair.mar, digits=6)) | PSR: $(round(summary_pair.psr, digits=6)) | DSR: $(round(summary_pair.dsr, digits=6))")


    # 3. Estratégia Sazonalidade
    println("\n" * "="^50)
    println("3️⃣  Executando Estratégia: SAZONALIDADE")
    println("="^50)
    cfg_seas = SeasonalityConfig()
    strategy_seas = Seasonality()
    res_seas = run_wfo(strategy_seas, copy(df_mom), cfg_seas, "Otimizada"; price_col=:Close)
    summary_seas = summarize_result(res_seas; n_trials=3)
    println("📅 Resultados por Split:")
    for m in summary_seas.split_metrics
        println("   Split $(m.split) (Período de Teste: $(m.test_dates[1]) até $(m.test_dates[2])):")
        println("      MAR: $(round(m.mar, digits=6)) | PSR: $(round(m.psr, digits=6)) | DSR: $(round(m.dsr, digits=6)) | Dias: $(m.n_days)")
        println("      📊 Parâmetro: Compra nos $(res_seas.best_params_history[m.split].k) meses com melhor performance histórica")
    end
    println("   📊 Resultado Geral (média de todos os splits):")
    println("      MAR: $(round(summary_seas.mar, digits=6)) | PSR: $(round(summary_seas.psr, digits=6)) | DSR: $(round(summary_seas.dsr, digits=6))")
    
    println("\n" * "="^75)
    println("✅ Execução de backtests concluída.")
end

# Executa a análise se o script for chamado diretamente
if abspath(PROGRAM_FILE) == @__FILE__
    run_all_strategies()
end