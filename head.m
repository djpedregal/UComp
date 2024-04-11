function y = head(x, nElem)
    if nargin < 2, nElem = 10; end
    if nElem > size(x, 1)
        y = x;
%         if nargout < 1
%             y
%         end
    else
        y = x(1 : nElem, :);
%         if nargout < 1
%             y
%         end
    end
end