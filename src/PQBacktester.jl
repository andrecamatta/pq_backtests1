module PQBacktester

# 1. Inclusão dos componentes do módulo
include("Config.jl")
include("Metrics.jl")
include("Costs.jl")
include("Split.jl")
include("Data.jl")
include("Strategies.jl")
include("Core.jl")

# 2. Importação e Reexportação (usando os submódulos)
using .Config
using .Metrics
using .Costs
using .Split
using .Data
using .Strategies
using .Core

# --- Exportações Públicas ---
# Config
export WFOConfig, MomentumConfig, PairsConfig, SeasonalityConfig
# Metrics
export sharpe, max_drawdown, cagr, mar, psr, dsr,
       weighted_mean_metric, bootstrap_ci, compound_sharpe, newey_west_sharpe
# Costs
export net_returns
# Split
export walk_forward_splits
# Data
export load_real_data, prepare_single_asset, prepare_pair
# Strategies
export Strategy, MomentumTS, PairsCoint, Seasonality, month_names
# Core
export RunResult, run_wfo, summarize_result

end # module PQBacktester