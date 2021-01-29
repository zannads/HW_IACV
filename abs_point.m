classdef abs_point
    %abs_point class to use well the points and not confuse them with lines
    
    properties
        params(3, 1) {mustBeReal}
        letter
    end
    
    methods
        function obj = abs_point( point_ , letter)
            %abs_point Construct an instance of this class
            %   coordinates and eventually a letter to name it in pictures
            if nargin == 1
                obj.params = point_;
                obj.letter = [];
            elseif nargin == 2
                obj.params = point_;
                obj.letter = letter;
            else
                obj.params = [0;0;1];
                obj.letter = [];
            end
        end
        
        function obj = sinqle_acq(obj)
            %acquire one point from the image 
            % you can choose multiples, but seqent from first are neglected
            figure(gcf);
            [x_, y_]=getpts;
            obj.params = [x_(1), y_(1), 1]';
        end
        
        function obj = draw(obj, varargin)
            %draw the point, eventually with its tag
            if( nargin >1 && strcmp( varargin{1} , "at") )
                figure(varargin{2});
            else
                figure(gcf);
            end
            hold on;
            plot( obj.params(1), obj.params(2),'.r','MarkerSize',12, 'LineWidth', 3);
            if( ~isempty(obj.letter) )
                text( obj.params(1), obj.params(2), obj.letter, 'FontSize', 20, 'Color', 'r')
            end
            hold off;
        end
        
        function outputArg = passing_line(obj,obj2)
            %passing_line given two points, which line goes by?
            outputArg = cross(obj.params, obj2.params);
            outputArg = outputArg/outputArg(3);
        end
        
        function obj = normalize (obj )
            %normalzie coordinates
            obj.params = obj.params ./ obj.params(3);
            
        end
        
        function obj = transform (obj, H)
            %transformation H is applied
            obj.params = H * obj.params ;
        end
     
    end
end

