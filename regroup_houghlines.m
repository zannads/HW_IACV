function grouped_lines = regroup_houghlines( lines )
%REGROUP_HOUGHLINES use output of Houghlines to regroup them how I want
%   The functions choose all the segments that belongs to the same line and
%   group them into a cell

idx = 1;
grouped_lines = cell(0);
while idx > 0
    % first line to look
    actual_theta = lines(1).theta;
    actual_rho = lines(1).rho;
    
    %look how many are equal
    jdx = 2;
    while  ( jdx <= length(lines) & actual_rho == lines(jdx).rho & lines(jdx).theta == actual_theta  ) %#ok<AND2>
        jdx = jdx+1;
    end
    
    %now that I know group the segments
    grouped_lines{idx} = lines( 1: min( jdx-1 , length( lines ) ) );
    
    % if not at the end
    if ( jdx < length(lines) )
        %remove already grouped lines
        lines = lines(jdx:end);
        idx = idx +1 ;
    else
        % you're done
        idx = 0;
    end
end


end

