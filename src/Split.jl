module Split
    using Dates
    export walk_forward_splits

    function month_add(d::Date, m::Int)
        y = year(d); mo = month(d) + m
        y += (mo - 1) ÷ 12
        mo = (mo - 1) % 12 + 1
        day_val = min(Dates.day(d), Dates.day(Dates.lastdayofmonth(Date(y, mo, 1))))
        return Date(y, mo, day_val)
    end

    function walk_forward_splits(dates::Vector{Date};
            train_months::Int=36, test_months::Int=12,
            embargo_days::Int=5, label_horizon_days::Int=0)
        @assert issorted(dates)
        res = Vector{NamedTuple{(:train_idx,:test_idx),Tuple{Vector{Int},Vector{Int}}}}()
        i_start = 1
        while true
            train_start_date = dates[i_start]
            train_end_date   = min(month_add(train_start_date, train_months)-Day(1), dates[end])
            test_start_date  = train_end_date + Day(1)
            test_end_date    = min(month_add(test_start_date, test_months)-Day(1), dates[end])
            if test_start_date > dates[end]
                break
            end
            train_idx = findall(d -> d >= train_start_date && d <= train_end_date - Day(label_horizon_days + embargo_days), dates)
            test_idx  = findall(d -> d >= test_start_date && d <= test_end_date, dates)
            if !isempty(train_idx) && !isempty(test_idx)
                push!(res, (train_idx=train_idx, test_idx=test_idx))
            end
            next_start_date = test_end_date + Day(embargo_days + 1)
            i_next = searchsortedfirst(dates, next_start_date)
            if i_next > length(dates)
                break
            end
            i_start = i_next
        end
        return res
    end
end