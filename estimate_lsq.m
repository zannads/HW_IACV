function request_object = estimate_lsq(inputArg1, varargin)
%ESTIMATE_LSQ least square estimation for my classes 
%   Perform a least square approach to find [x1; x2; 1]

%build the matrix 
% should be l' * x = 0 or x' l = 0
% anyway params goes horizontal
% as many lines as many object 
 A = zeros(length( inputArg1 ), 3);
    for idx = 1:length(inputArg1 )
       A(idx, :) = inputArg1(idx).params';
    end
    % first two columns are multplied by unknowns x1 and x2
    a = A(:, 1:2);
    % last column is multiplied by 1 thus it goes to right end side
    b = -A(:, end);
    % you should know better
    x = pinv(a)*b;

    % if I had specified the class I cast it into
    % if is a vanishing point I save it's name too 
if ( strcmp( varargin{1}, 'vp' ) )
    request_object = abs_point( [x; 1], varargin{2} );
else %if( strcmp( inputArg2, 'linf') ) 
    request_object = abs_line( [x; 1] );
end

end