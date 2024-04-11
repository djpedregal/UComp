function yts = ts(y, start, frequency)
% yts = ts(y, start, frequency)
%
% y:         time series
% start:     fecha inicio [Y M D]
% frequency: number of observations per cycle
%
    if nargin < 2
        start = [1990 1 31];
    end
    if length(start) == 1
        start = [start 1 31];
    end
    start = datetime(start);
    if nargin < 3
        frequency = 12;
    end
    n = size(y, 1);
    if frequency == 12
        t = (start : calmonths(1) : (start + calmonths(n - 1)))';
    elseif frequency == 4
        t = (start : calquarters(1) : (start + calquarters(n - 1)))';
    elseif frequency == 1
        t = (start : calyears(1) : (start + calyears(n - 1)))';
    elseif frequency == 24
        t = (start : hours(1) : (start + hours(n - 1)))';
    end
    yts = timetable(y, 'VariableNames', {'Data'}, 'RowTimes', t);
end
