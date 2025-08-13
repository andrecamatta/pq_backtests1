module Data
    using Dates, DataFrames, YFinance
    export load_real_data, prepare_single_asset, prepare_pair

    function load_real_data(symbol::String; start_date::Date=Date(2018,1,1), end_date::Date=today())
        try
            println("   Carregando $symbol...")
            start_str = Dates.format(start_date, "yyyy-mm-dd")
            end_str = Dates.format(end_date, "yyyy-mm-dd")
            
            data = get_prices(symbol, startdt=start_str, enddt=end_str)
            
            if isempty(data)
                @warn "Dados vazios para $symbol"
                return DataFrame()
            end
            
            df = DataFrame()
            df[!, :Date] = Date.(data["timestamp"])
            df[!, :Close] = Float64.(data["adjclose"])
            
            sort!(df, :Date)
            dropmissing!(df)
            filter!(:Close => x -> x > 0, df)
            
            return df
        catch e
            @warn "Erro carregando $symbol: $e"
            return DataFrame()
        end
    end

    function prepare_single_asset(df::DataFrame)
        @assert "Date" in names(df) && "Close" in names(df)
        rename!(df, "Date" => :Date, "Close" => :Close)
        sort!(df, :Date)
        return df
    end

    function prepare_pair(df1::DataFrame, df2::DataFrame)
        for c in (df1, df2)
            @assert "Date" in names(c) && "Close" in names(c)
            rename!(c, "Date" => :Date, "Close" => :Close)
            sort!(c, :Date)
        end
        df = innerjoin(rename(df1, :Close=>:Close1), rename(df2, :Close=>:Close2), on=:Date)
        return df
    end
end