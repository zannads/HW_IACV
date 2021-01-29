cd('/Users/denniszanutto/Documents/GitHub/HW_IACV');
im_name = 'Image - Castello di Miramare.jpg';
im=imread(im_name);

figure;
imshow(im);

% create gray scale image and create imref object so that the new plane
% will be in scaled coordinates, between 0 and 1 along x and 0 and
% 2796/3968 along y
im_g = rgb2gray(im);
xWorldLimits = [0, 1];
yWorldLimits = [0, 0.75];
WR =imref2d(size(im_g),xWorldLimits,yWorldLimits) ;
H_resize = [WR.PixelExtentInWorldX, 0, 0;
    0, WR.PixelExtentInWorldY, 0;
    0,              0,         1];
% number of different feature lines 5 on plane PI, 1 vertical
num_lines = 6;
%% IMAGE PROCESSING 
% corner detection
corner = detectHarrisFeatures(im_g);
figure; imshow(im);
hold on;
plot(corner.selectStrongest(100));
%%
% I define some masks of binary images for selecting the area where I'm
% interset in to finf lines
edgs = edge(im_g,'canny');
%maximum number of lines for every time I run Hough
limit_n_lines = 5;

%start masks
m1 = zeros(size(im, 1:2));
m1(1:2763, 1:1530, 1) = ones( 2763, 1530);

m2 = zeros([size(im, 1:2), 2]);
m2(2000:2800, 1028:1832, 1 ) = ones(801, 805);
m2(1000:1600, 1130:1650, 2 ) = ones(601, 521);

m3 = zeros(size(im, 1:2));
m3(:, 1434:2434 ) = ones(2976, 1001);

m4 = zeros(size(im, 1:2));
m4(:, 1900:end ) = ones(2976, 2069);

m5 = zeros( [size(im, 1:2), 2]);
m5(:, 2500:3200, 1 ) = ones(2976, 701);
m5(2500:2700, 2650:3300, 2) = ones( 201, 651);

m6 = zeros(size(im, 1:2));
m6(1:2500, :) = ones( 2500, 3968);
%
% figure; imshow(im);
%  figure;
%  imshow(edgs&m5(:,:,2));
%

% cells of parameters for the different facade 1 to 5 are the facade, 6 is
% vertical direction
title_select = ["1", "2", "3", "4 & 6", "5", "vertical" ];

canny = {[0.01, 0.2], [0.02, 0.2], [0.01, 0.1], [0.01, 0.2], [0.005, 0.05], [0.01, 0.2], };
mask = { m1, m2, m3, m4, m5, m6};
rho_res = {1, 1, 1, 1, 2, 1};
theta_res = {80:0.5:89, -60:0.5:-30, -80:0.5:-60, -90:0.5:-80, 30:0.5:80, -20:0.5:20};
fill_gap_length = 40; %default 20
min_length = {50, 30, 50, 50, 50, 50};
%%
% start of selected lines, the algorithm does this:
% for every direction from 1 to 6, 
%   uploads the masks
%   runs canny 
%   Hough lines detection (output is an array of segments)
%   regroup the segments in the same direction (almost a line)
%   displays this set
%   asks for confirmation
%   if line is confirmed, 
%           transforms from Hough params to my line class
%           change of coordinates

set_plines = cell( num_lines, 1 );
idx = 1;
while idx <= num_lines
    saved = 0;
    
    level = size(mask{idx}, 3);
    for sub = 1:level
        if(level>1)
            mask_i  = mask{idx}(:,:, sub);
        else
            mask_i = mask{idx};
        end
        edgs = edge(im_g, 'canny', canny{idx});
        edgs = edgs&mask_i;
        figure; imshow(edgs, WR);
        
        [H,theta,rho] = hough(edgs, 'RhoResolution', rho_res{idx}, 'Theta', theta_res{idx});
        P = houghpeaks(H, ceil(limit_n_lines/level), 'threshold', ceil(0.3*max(H(:))));
        lines = houghlines(edgs,theta,rho,P,'FillGap',fill_gap_length,'MinLength',min_length{idx});
        
        % If you also want colormap
        % figure;
        % imshow(imadjust(rescale(H)),[],...
        %        'XData',theta,...
        %        'YData',rho,...
        %        'InitialMagnification','fit');
        % xlabel('\theta (degrees)')
        % ylabel('\rho')
        % axis on
        % axis normal
        % hold on
        % colormap(gca,hot)
        %     x = theta(P(:,2));
        %     y = rho(P(:,1));
        %     plot(x,y,'s','color','black');
        
        glines = regroup_houghlines( lines );
        
        f_h = figure; imshow(im); hold on;
        
        k = 1;
        while k <= length( glines )
            title( strcat( "Select lines parallel to ", title_select(idx) ) );
            
            line_ = glines{k};
            
            figure(f_h);hold on;
            for m = 1:length( line_ )
                xy = [line_(m).point1; line_(m).point2];
                plot(xy(:,1), xy(:,2),'LineWidth',2,'Color','green');
                
                % Plot beginnings and ends of lines
                plot(xy(1,1), xy(1,2), 'x','LineWidth',2,'Color','yellow');
                plot(xy(2,1), xy(2,2), 'x','LineWidth',2,'Color','red');
                
            end
            
            answer = questdlg('Keep the line?', ...
                'Line select', ...
                'Yes', 'Yes | plane PI', 'No', 'No');
            
            switch answer
                case 'Yes'
                    save_it  = 1;
                case 'Yes | plane PI'
                    save_it = 1;
                    disp( strcat( "Line on PI, set = ", num2str( idx ), "number: ", num2str( saved + 1) ) );
                case 'No'
                    save_it = 0;
                    %disp('not saved');
            end
            
            if (save_it)
                %transform it into myclass
                lines_to_transform = glines{k}(1);
                a = cos( lines_to_transform.theta * pi/180 );
                b = sin( lines_to_transform.theta *pi/180);
                c =  -lines_to_transform.rho ;
                general_line = abs_line ( [a; b; c]/ c );
                %I want them in unitary coordinates
                general_line = general_line.transform( H_resize );
                %add to te ones already saved
                set_plines{idx} = [set_plines{idx}; general_line];
                saved = saved+1;
            end
            
            k = k+1;
        end
        
        %     % check if it's at least a couple for every set of plines
        %     if ( size( set_plines{idx}, 1 ) < 2 )
        %         %manually select
        %         missing = 2 - size( set_plines{idx}, 1 );
        %         count = 1;
        %         %acquire from image missing lines
        %         toget_lines(missing,1) = abs_line();
        %         while (count <=missing)
        %             figure(gcf);
        %             toget_lines(count) = toget_lines(count).acquire_from_image();
        %
        %             % again in unitary coordinates
        %             toget_lines(count) = toget_lines(count).transform( H_resize );
        %
        %             count = count +1;
        %         end
        %
        %         %add lines acquired
        %         set_plines{idx} = [set_plines{idx}; toget_lines];
        %         clear toget_lines;
        %     end
    end
    idx = idx+1;
end
hold off;

% when i need to save them for geometry 
%    save('parallel_lines.mat', 'set_plines');

%% GEOMETRY


% If i don't run image processing first 
old_lines = load('parallel_lines.mat');
set_plines = old_lines.set_plines;
clear old_lines;

%first of all I have enough lines for vanishing points
set_vpoints(num_lines, 1) = abs_point();
% since lines are already in unitary coordinates they should be too
for idx = 1: length(set_plines)
    point_name = strcat("V", num2str(idx));
    if( size(set_plines{idx}, 1) < 4 )  
        % I choose that if I have 3 lines I still neglects one, errors are
        % too big
        set_vpoints(idx) = abs_point( cross( set_plines{idx}(1).params, set_plines{idx}(2).params ),  point_name);
        set_vpoints(idx) = set_vpoints(idx).normalize();
    else
        % it doen't improve a lot, is just a show off
        set_vpoints(idx) = estimate_lsq( set_plines{idx}, 'vp', point_name );
        set_vpoints(idx) = set_vpoints(idx).normalize();
    end
end

%Now i define the lines on plane PI
pi_lines = { set_plines{1}(1), set_plines{2}(3), set_plines{3}(2), set_plines{4}(4), set_plines{5}(1), set_plines{4}(1)};

% point between 5 and vertical that i know
point_ = cross( pi_lines{5}.params, [-1.525; 0.2084; 1] );

pi_lines{6} = abs_line( cross( point_, set_vpoints(4).params ) );

pi_lines{1} = pi_lines{1}.reset_point( [1; 0; 0], 1);
pi_lines{1} = pi_lines{1}.reset_point( pi_lines{2}.params, 2);
pi_lines{2} = pi_lines{2}.reset_point( pi_lines{1}.params, 1);
pi_lines{2} = pi_lines{2}.reset_point( [1; 0; -0.4], 2);

pi_lines{3} = pi_lines{3}.reset_point( [1; 0; -0.43], 1);
pi_lines{3} = pi_lines{3}.reset_point( [1; 0; -0.56], 2);

pi_lines{4} = pi_lines{4}.reset_point( [1; 0; -0.55], 1);
pi_lines{4} = pi_lines{4}.reset_point( pi_lines{5}.params, 2);
pi_lines{5} = pi_lines{5}.reset_point( pi_lines{4}.params, 1);
pi_lines{5} = pi_lines{5}.reset_point( pi_lines{6}.params, 2);
pi_lines{6} = pi_lines{6}.reset_point( pi_lines{5}.params, 1);
pi_lines{6} = pi_lines{6}.reset_point( [1; 0; -1], 2);
clear point_

reference_point = abs_point( pi_lines{1}.p2, "o_{\pi}");
% end of definition of plane pi

f_h = figure;
imshow(im);
hold on

for idx = 1:length( pi_lines)
    lines_to_draw = pi_lines{idx};
    %I'm plotting without WR this time, so go back to pixel coordinates
    lines_to_draw = lines_to_draw.transform( inv(H_resize) );   
    lines_to_draw.draw("at", f_h, 'Color', 'y', 'LineWidth', 3);
end
t = reference_point;
t = t.transform( inv(H_resize) );
t.draw("at", f_h);
clear lines_to_draw t;

%  draw them all again
f_h = figure;
imshow(im);
hold on
colors = 'rgbcmykwrgbcmykw';
for idx = 1:length( set_plines)
    lines_to_draw = set_plines{idx};
    for jdx = 1:length( lines_to_draw)
        lines_to_draw(jdx) = lines_to_draw(jdx).transform( inv(H_resize) );
        lines_to_draw(jdx).draw("at", f_h, 'insideBorder', size(im, 1:2), 'Color', colors(idx) );
    end
end
clear lines_to_draw colors;
for idx = 1: length(set_plines)
    p_ = set_vpoints(idx);
    p_ = p_.transform( inv(H_resize) );
    p_.draw();
end
clear p_;

linf_1 = estimate_lsq( set_vpoints(1:5), 'linf', 'linf_\pi' );

% naive way to find line at the infinity 
% linf_1 = abs_line( cross( set_vpoints(1).params, set_vpoints(4).params ), ...
%     set_vpoints(1).params, set_vpoints(4).params );
linf_1 = linf_1.normalize();
%just to draw it in the same ref as he others
linf_draw = linf_1.transform( inv(H_resize) );
linf_draw.draw("at", f_h);
clear linf_draw

Har = [1,0,0; 0,1,0; linf_1.params'];
%% Warping 1
tform = projective2d(Har');
[im_ar, WR_ar] = imwarp(im, WR, tform);

f_h = figure;
imshow(im_ar, WR_ar ); title("affine rect");
% i need to transform also the yellow lines
pi_lines_ar = pi_lines;
for idx = 1:6
    pi_lines_ar{idx} = pi_lines_ar{idx}.transform( Har );
    pi_lines_ar{idx} = pi_lines_ar{idx}.normalize();
    pi_lines_ar{idx}.draw( "at", f_h, 'Color', 'y', 'LineWidth', 3);
end
reference_point_ar  = reference_point;
reference_point_ar = reference_point_ar.transform( Har );
reference_point_ar = reference_point_ar.normalize();

% I also need the set of parallell lines for orthogonality
f_h = figure;
imshow(im_ar, WR_ar ); title("affine rect");
colors = 'rgbcmykwrgbcmykw';
set_plines_ar = set_plines;
for idx = 1:length( set_plines_ar )
    line_to_t = set_plines_ar{idx};
    
    for jdx = 1:length( line_to_t )
        line_to_t(jdx) = line_to_t(jdx).transform( Har );
        line_to_t(jdx) = line_to_t(jdx).normalize();
        line_to_t(jdx).draw("at", f_h, 'insideBorder', size(im_ar, 1:2), 'Color', colors(idx) );
    end
    
    set_plines_ar{idx} = line_to_t;
end
clear colors line_to_t;
%% Warping intermediate
% unfortunately images become too big for my computer, I rescale them and I
% will remember later of course
s = 0.25;
rescale = [s, 0, 0; 0, s, 0; 0, 0, 1];

tform = projective2d(rescale');
[im_ar_resc, WR_ar_resc] = imwarp(im_ar, WR_ar, tform);
f_h = figure;
imshow(im_ar_resc, WR_ar_resc ); title("affine rect");
% i need to transform also the yellow lines
pi_lines_ar_resc = pi_lines_ar;
for idx = 1:6
    pi_lines_ar_resc{idx} = pi_lines_ar_resc{idx}.transform( rescale );
    pi_lines_ar_resc{idx} = pi_lines_ar_resc{idx}.normalize();
    pi_lines_ar_resc{idx}.draw( "at", f_h, 'Color', 'y', 'LineWidth', 3);
end
reference_point_ar_resc  = reference_point_ar;
reference_point_ar_resc = reference_point_ar_resc.transform( rescale );
reference_point_ar_resc = reference_point_ar_resc.normalize();

% I also need the set of parallell lines for orthogonality

f_h = figure;
imshow(im_ar_resc, WR_ar_resc ); title("affine rect");
set_plines_ar_resc = set_plines_ar;
colors = 'rgbcmykwrgbcmykw';
for idx = 1:length( set_plines_ar_resc )
    line_to_t = set_plines_ar_resc{idx};
    
    for jdx = 1:length( line_to_t )
        line_to_t(jdx) = line_to_t(jdx).transform( rescale );
        line_to_t(jdx) = line_to_t(jdx).normalize();
        line_to_t(jdx).draw("at", f_h, 'insideBorder', size(im_ar_resc, 1:2), 'Color', colors(idx) );
    end
    
    set_plines_ar_resc{idx} = line_to_t;
end
clear line_to_t colors;
%%
% then I can use {1} and {2} and {5} and {4}
numConstraints = 2;
count = 1;
% select pairs of orthogonal segments
A = zeros(numConstraints,3);
orthogonal_l(2*numConstraints, 1) = abs_line( );
orthogonal_l(1) = set_plines_ar_resc{1}(1);
orthogonal_l(2) = set_plines_ar_resc{2}(1);
orthogonal_l(3) = set_plines_ar_resc{4}(1);
orthogonal_l(4) = set_plines_ar_resc{5}(1);


while (count <=numConstraints)
    % each pair of orthogonal lines gives rise to a constraint on s
    % [l(1)*m(1),l(1)*m(2)+l(2)*m(1), l(2)*m(2)]*s = 0
    % store the constraints in a matrix A
    l = orthogonal_l(2*count-1).params(1:2);
    m = orthogonal_l(2*count).params(1:2);
    A(count,:) = [l(1)*m(1), l(1)*m(2)+l(2)*m(1), l(2)*m(2)];
    count = count+1;
end
clear count numConstraints orthogonal_l l m

%S = [x(1) x(2); x(2) 1];
[~,~,v] = svd(A);
s = v(:,end); %[s11,s12,s22];
S = [s(1),s(2); s(2),s(3)];

%imDCCP = [S,zeros(2,1); zeros(1,3)]; % the image of the circular points
[U,D,V] = svd(S);
A = U*sqrt(D)*V';
H = eye(3);
H(1,1) = A(1,1);
H(1,2) = A(1,2);
H(2,1) = A(2,1);
H(2,2) = A(2,2);

Hrect = inv(H);
%%
tform = projective2d(Hrect');
[im_rectified, WR_rectified] = imwarp(im_ar_resc, WR_ar_resc, tform);

f_h = figure;
imshow( im_rectified, WR_rectified );

% i need to transform also the yellow lines
pi_lines_rect = pi_lines_ar_resc;
for idx = 1:6
    pi_lines_rect{idx} = pi_lines_rect{idx}.transform( Hrect );
    pi_lines_rect{idx} = pi_lines_rect{idx}.normalize();
    pi_lines_rect{idx}.draw( "at", f_h, 'Color', 'y', 'LineWidth', 3);
end

reference_point_rect  = reference_point_ar_resc;
reference_point_rect = reference_point_rect.transform(Hrect);
reference_point_rect = reference_point_rect.normalize();
%% calibration

% K =
%
% [ fx,  0, u0]
% [  0, fy, v0]
% [  0,  0,  1]

% w = inv(K*K')

% w = [ a^2, 0, -u0*a^2; 
%       0, 1, -v0; 
%       -u0*a^2, -v0, fy^2+a^2u0^2+vo^2]
% w = [ w11, 0,    w13;  
%       0, 1, w23;   
%       w13,    w23,    w33]

% w = 4 elements, w21 and 22 are 0; it's symmetric
% to compute w I need 4 perpendicular vanishing point
% this means: vj w vi = 0
% vl1 + vv
% vl2 + vv
% vl3 + vv
% vl4 + vv
numConstraints = 4;
select = {[1, 2], [1, 6], [4, 6], [4, 5]};

A = zeros(numConstraints, numConstraints+1);
%A (1, :) = v1_1*v2_1*w11 + (v_1_1*w13 + v_1_2*w23 + v_1_3*w33) + v_2_2*(v_1_2 + v_1_3*w23) + v_1_3*v_2_1*w13
%A (1, :) = [v1_1*v2_1, v_1_1 + v_2_1, v_1_2+v_2_2 , v_1_3*v_2_3 , v_1_2*v_2_2]; % this for every couple
for idx = 1:length(select)
    v_1 = set_vpoints( select{idx}(1) ).params;
    v_2 = set_vpoints( select{idx}(2) ).params;
%     % for having it in pixel coordinates uncomment this part!!!
%     v_1 =  set_vpoints( select{idx}(1) ).transform( inv(H_resize) );
%     v_2 = set_vpoints( select{idx}(2) ).transform( inv(H_resize) );
%     v_1 = v_1.params/v_1.params(3);
%     v_2 = v_2.params/v_2.params(3);
%     %
    
    A(idx, :) = [v_1(1)*v_2(1),  v_1(1)+v_2(1),  v_1(2)+v_2(2),  1,  v_1(2)*v_2(2)];
end

% solve for x = [w11, w13, w23, w33];
x = linsolve( A(:, 1:end-1), -(A(:,end)) );

IAC = [x(1),    0,     x(2);
    0,       1,     x(3);
    x(2), x(3), x(4)]

a = sqrt(IAC(1,1));
u0 = -IAC(1,3)/(a^2);
v0 = -IAC(2,3);
fy = sqrt(IAC(3,3) - (a^2)*(u0^2) - (v0^2));
fx = fy /a;

% build K
K = [fx, 0, u0; 0, fy, v0; 0, 0, 1]

w = inv(K*K');
% w = w/w(5)
% %alternative using cholesky
% K_ = chol( inv(IAC) );
% K_ = K_/K_(9)
% w_ = inv(K_*K_');
% w_ = w_/w_(5)
clear a fx fy u0 v0 IAC x A select numConstraints v_1 v_2
%% orientation
% I need to find the Homography from real plane to image
% I already did
H_omog = inv( Hrect*rescale*Har );
% Find rotation and translation matrix from H_omog
R = K \ H_omog;
lambda = 1 / norm(K \ H_omog(:,1));
R = R .* lambda;
i_pi = R(:,1);
j_pi = R(:,2);
o_pi = R(:,3);

% third direction is the cross product since directions should be
% orthogonal
R = [ i_pi, j_pi, cross(i_pi,j_pi)];

% matrix from point on the plane to point in the world 
world = [R, o_pi;
    zeros(1, 3), 1];

% Find camera rotation and translation wrt to the ref of the plane
camera_rotation = inv(R)
camera_position = -R\o_pi

% 3d plot plane wrt to camera
figure;
for idx = 1:2:2*length(pi_lines_rect)
    % go to world coordinates for every point in the plane wrt to camera
    where = ceil(idx/2);
    point = [ (pi_lines_rect{where}.p1(1:2) ); 0; pi_lines_rect{where}.p1(3)] ;
    point_ = [ (pi_lines_rect{where}.p2(1:2) ); 0; pi_lines_rect{where}.p2(3)] ;
    p_w(:, 1)=  world*point;
    p_w(:, 2) = world*point_;
    p_w = p_w./p_w(end,:);
    plot3(p_w(1,1:2), p_w(2,1:2), p_w(3,1:2), 'y', 'LineWidth', 3);
    hold on;
end
pose = rigid3d( eye(3), zeros(1, 3) );
plotCamera('AbsolutePose',pose,'Opacity',0, 'Size', 0.01)
grid on
axis equal

%3d plot camera wrt to plane
figure;
ref = [reference_point_rect.params(1:2); 0; 0];
for idx = 1:2:2*length(pi_lines_rect)
    where = ceil(idx/2);
    point = [ (pi_lines_rect{where}.p1(1:2) ); 0; pi_lines_rect{where}.p1(3)] -ref;
    point_ = [ (pi_lines_rect{where}.p2(1:2) ); 0; pi_lines_rect{where}.p2(3)] -ref;
    p_w = [point, point_] ;
    p_w = p_w./p_w(end,:);
    plot3(p_w(1,1:2), p_w(2,1:2), p_w(3,1:2), 'y', 'LineWidth', 3);
    hold on;
    plot3(p_w(1,1), p_w(2,1), -1, 'y', 'LineWidth', 1);
    plot3(p_w(1,2), p_w(2,2), -1, 'y', 'LineWidth', 3);
end
% unfortunately rotation is not rigid, I cant' use plotCamera
plot3( camera_position(1)-ref(1), camera_position(2)-ref(2), camera_position(3) , 'or');
plot3( 0, 0, 0 , '*r');
text( 0.05,0.05,0, 'O_\pi', 'FontSize', 20, 'Color', 'r')
grid on
axis equal

clear ref where point point_ p_w
%% Reconstruction vertical
% just select which one between 1 and 4 and the script will do the rest
% automatically
facade = 1;

% lina the infinity for the required plane 
linf_vert = abs_line( cross( set_vpoints(facade).params, set_vpoints(6).params ), set_vpoints(facade).params, set_vpoints(6).params  );
linf_vert = linf_vert.normalize();

% find I'= I_p and J'
syms u1 u2 u3
assume(u1 ~= 0);
assume(u2 ~= 0);
eqns = [ linf_vert.params(1)*u1 + linf_vert.params(2)*u2 + 1 == 0, w(1,1)*u1^2 +  w(2,2)*u2^2 + 2*w(1,3)*u1 + 2*w(2,3)*u2 + w(3,3) ==0];
S = vpasolve(eqns, [u1 u2 ]);
u_1 = double(S.u1);
u_2 = double(S.u2);
I_ = [u_1(1); u_2(1); 1];
J_ =  [u_1(2); u_2(2); 1];
% check it's correct ( I had problems) 
if ( linf_vert.params'*I_ > 1e-5)
    disp( "something bugged");
end


% conic dual to the circular points = I'J'^T+J'I'T
DC_inf_ = I_*J_.' + J_*I_.';

% Professor said that I should extract it from SVD 
%[~, ~, Hr^T] = svd(DC_inf_)
% [U, A, V] = svd(DC_inf_);
% Q = sqrt(A);
% Q(3,3) = 1;
% Hr_vert = U*Q;
% well I couldn't so 
U = [real(I_), imag(I_), [0;0;1] ];
Hr_vert = inv(U);

% warp something again
tform = projective2d( Hr_vert' );
[im_vert, WR_vert] = imwarp(im, WR, tform);
f_h = figure;
imshow(im_vert,WR_vert);

% i need to transform also the yellow lines
pi_lines_vert = pi_lines;
for idx = 1:6
    pi_lines_vert{idx} = pi_lines_vert{idx}.transform( Hr_vert );
    pi_lines_vert{idx} = pi_lines_vert{idx}.normalize();
    pi_lines_vert{idx}.draw( "at", f_h, 'Color', 'y', 'LineWidth', 3);
end

% also the old set, even if actually I'm interested oinly in verticals 
set_plines_vert = set_plines;
for idx = 1:length( set_plines_vert )
    line_to_t = set_plines_vert{idx};
    
    for jdx = 1:length( line_to_t )
        line_to_t(jdx) = line_to_t(jdx).transform( Hr_vert );
        line_to_t(jdx) = line_to_t(jdx).normalize();
    end
    
    set_plines_vert{idx} = line_to_t;
end
clear line_to_t colors;

% I know vertical lines end up rotated
raddr = set_plines_vert{6}.params;
raddr = raddr/raddr(3);
theta = -atan( -raddr(1)/raddr(2) ) - pi/2;

H_rot = [cos(theta), -sin(theta), 0; 
            sin(theta), cos(theta), 0;
            0, 0, 1];
% rotate, maybe there is something faster, but I don't want to look for it
tform = projective2d( H_rot' );
[im_vert, WR_vert] = imwarp(im_vert, WR_vert, tform);

f_h = figure;
imshow(im_vert,WR_vert);
for idx = 1:6
    pi_lines_vert{idx} = pi_lines_vert{idx}.transform( H_rot );
    pi_lines_vert{idx} = pi_lines_vert{idx}.normalize();
    pi_lines_vert{idx}.draw( "at", f_h, 'Color', 'y', 'LineWidth', 3);
end

%tadaaaan 
% Dennis Zanutto 
% 28 / 01 / 2021