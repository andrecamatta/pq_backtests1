module Costs
    export net_returns

    function net_returns(gross::AbstractVector, pos::AbstractVector; commission=0.0, slippage=0.0)
        N = length(gross)
        @assert length(pos) == N "gross and pos must align"
        cost_per_unit = commission + slippage
        Δpos = [0.0; abs.(diff(pos))]
        costs = cost_per_unit .* Δpos
        return gross .- costs
    end
end