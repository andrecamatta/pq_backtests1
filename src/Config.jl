module Config
    using Dates
    export WFOConfig, MomentumConfig, PairsConfig, SeasonalityConfig

    Base.@kwdef mutable struct WFOConfig
        train_months::Int = 36
        test_months::Int  = 12
        embargo_days::Int = 5          
        label_horizon_days::Int = 0    
        rf_annual::Float64 = 0.0       # Sem risk-free, foco em alpha vs benchmark
        trading_days::Int = 252
        commission::Float64 = 0.0005   
        slippage::Float64  = 0.0005    
    end

    # Configurações otimizadas por estratégia
    function MomentumConfig()
        return WFOConfig(
            train_months=24,    # Suficiente para MA de 200 dias
            test_months=6,      # Rebalance semestral
            embargo_days=3,     # Estratégia de médio prazo
            commission=0.0005,
            slippage=0.0005
        )
    end

    function PairsConfig()
        return WFOConfig(
            train_months=36,    # Mais dados para estimar β robustamente
            test_months=12,     # Permite janela móvel maior (80 dias)
            embargo_days=5,     # Mean reversion precisa de mais tempo
            commission=0.0005,
            slippage=0.0005
        )
    end

    function SeasonalityConfig()
        return WFOConfig(
            train_months=24,    # Mais histórico para capturar padrões sazonais consistentes
            test_months=12,     # Períodos longos para capturar ciclos sazonais completos
            embargo_days=2,     # Estratégia baseada apenas no mês
            commission=0.0005,
            slippage=0.0005
        )
    end
end