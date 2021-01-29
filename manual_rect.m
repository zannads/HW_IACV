%% manually select mode

figure(5);
imshow(im, WR);

numConstraints = 4; %2 couples of lines
count = 1;
%v = zeros(3, numConstraints);

parallel_l(numConstraints, 1) = abs_line( );

while (count <=numConstraints)
    figure(gcf);
    title(['Draw ', num2str(numConstraints),' pairs of parallel segments: step ',num2str(count) ]);
    
    parallel_l(count) = parallel_l(count).acquire_from_image();
    parallel_l(count) = parallel_l(count).normalize();
    count = count +1;
end

v1 = abs_point( parallel_l(1).intersection(parallel_l(2) ), 'v1' );
v1 = v1.normalize();
v2 = abs_point( intersection(parallel_l(3), parallel_l(4) ) , 'v2');
v2 = v2.normalize();

v1.draw();
v2.draw();

linf_1 = abs_line (cross(v1.params, v2.params), v1.params, v2.params );
linf_1 = linf_1.normalize();
linf_1.params = linf_1.params; % .* [x; x; 1];
linf_1.draw();

Har = [1,0,0; 0,1,0; linf_1.params'];

tform = projective2d(Har');
[im_ar, WR_ar] = imwarp(im, WR, tform);
figure;
imshow(im_ar, WR_ar); title("affine rect");

count = 1;
numConstraints = 2;
% select pairs of orthogonal segments
A = zeros(numConstraints,3);
orthogonal_l(2*numConstraints, 1) = abs_line( );

while (count <=numConstraints)
    figure(gcf);
    title(['Draw ', num2str(numConstraints),' pairs of orthogonal segments: step ',num2str(count) ]);
    col = 'rgbcmykwrgbcmykw';
    %     segment1 = drawline('Color',col(count));
    %     segment2 = drawline('Color',col(count));
    %
    %     l = segToLine(segment1.Position);
    %     m = segToLine(segment2.Position);
    orthogonal_l(2*count-1) = orthogonal_l(count).acquire_from_image();
    % orthogonal_l(2*count-1) = orthogonal_l(count).normalize();
    orthogonal_l(2*count) = orthogonal_l(count).acquire_from_image();
    % orthogonal_l(2*count) = orthogonal_l(count).normalize();
    % each pair of orthogonal lines gives rise to a constraint on s
    % [l(1)*m(1),l(1)*m(2)+l(2)*m(1), l(2)*m(2)]*s = 0
    % store the constraints in a matrix A
    l = orthogonal_l(2*count-1).params(1:2);
    m = orthogonal_l(2*count).params(1:2);
    A(count,:) = [l(1)*m(1),l(1)*m(2)+l(2)*m(1), l(2)*m(2)];
    count = count+1;
end

%S = [x(1) x(2); x(2) 1];
[~,~,v] = svd(A);
s = v(:,end); %[s11,s12,s22];
S = [s(1),s(2); s(2),s(3)];

imDCCP = [S,zeros(2,1); zeros(1,3)]; % the image of the circular points
[U,D,V] = svd(S);
A = U*sqrt(D)*V';
H = eye(3);
H(1,1) = A(1,1);
H(1,2) = A(1,2);
H(2,1) = A(2,1);
H(2,2) = A(2,2);

Hrect = inv(H);
Cinfty = [eye(2),zeros(2,1);zeros(1,3)];

tform = projective2d(Hrect');
im_ar = imwarp(im,tform);

figure;
imshow(im_ar);

% while i < 4
% segmento i
% trasforma in linea i
% disegna linea i
%%%dA migliorare perchè linee non proprio parallele

% vpoint linea 1, 2
% vpoint line 3, 4

% horizone line

%construct HAr
% apply rectification

% while i < 2
% segmento i
% trasforma in linea i
% disegna linea i
%%%dA migliorare perchè linee non proprio perpendicolari

% construct Haffintity
% apply Haffinity
%%
% r = 0.5;
% movingPoints = [0, 0;
%                 0, r;
%                 r, 0;
%                 r, r];
%
% d1 = cross( pi_lines{1}.params, [0;0;1] );
% d1 = d1/norm(d1);
% d2 = cross( pi_lines{2}.params, [0;0;1] );
% d2 = d2/norm(d2);
% x_1 = abs_point( reference_point.params + r * d1 , "x_1" );
% x_2 = abs_point( reference_point.params + r * d2 , "x_2" );
% l3 = cross( x_1.params, set_vpoints(2).params );
% l4 = cross( x_2.params, set_vpoints(1).params );
% x_3 = abs_point( cross( l3, l4), "x_3" );
% x_3 = x_3.normalize();
% image_points_ref = [reference_point, x_2; x_1, x_3];
% fixedPoints = [reference_point.params(1:2)';
%                x_1.params(1:2)';
%                x_2.params(1:2)';
%                x_3.params(1:2)'];
%
% f_h = figure;
% imshow(im, WR);
% for idx = 1:4
%     image_points_ref(idx).draw("at", f_h);
% end
%
% %Find transformation between world points and image points
% H_omog = fitgeotrans(movingPoints, fixedPoints, 'projective');
% H_omog = H_omog.T';

%%
% too bright, try to correct exposure
%let's see brightness
% figure; histogram(im(:,:,1), 0:255, 'DisplayStyle', 'stairs'); title('red intensity histogram'); axis tight;
% figure; histogram(im(:,:,2), 0:255, 'DisplayStyle', 'stairs'); title('green intensity histogram'); axis tight;
% figure; histogram(im(:,:,3), 0:255, 'DisplayStyle', 'stairs'); title('blue intensity histogram'); axis tight;
% %%
% h_red = hist(reshape(im(:,:,1), 1, []), 0: 255); %#ok<*HIST>
% figure; stairs(0: 255, h_red), title('Red intensity histogram');
% axis tight
%
% h_green = hist(reshape(im(:,:,2), 1, []), 0: 255); %#ok<*HIST>
% figure; stairs(0: 255, h_green), title('Green intensity histogram');
% axis tight
%
% h_blue = hist(reshape(im(:,:,3), 1, []), 0: 255); %#ok<*HIST>
% figure; stairs(0: 255, h_blue), title('Blue intensity histogram');
% axis tight
%
% im_exp = histeq(im);
%
%
% h_red = hist(reshape(im_exp(:,:,1), 1, []), 0: 255); %#ok<*HIST>
% figure; stairs(0: 255, h_red), title('Red intensity histogram redist');
% axis tight
%
% h_green = hist(reshape(im_exp(:,:,2), 1, []), 0: 255); %#ok<*HIST>
% figure; stairs(0: 255, h_green), title('Green intensity histogram redist');
% axis tight
%
% h_blue = hist(reshape(im_exp(:,:,3), 1, []), 0: 255); %#ok<*HIST>
% figure; stairs(0: 255, h_blue), title('Blue intensity histogram redist ');
% axis tight
%
% imshow(im_exp);
%
%
% figure;imshow([im_g, rgb2gray(im)]);
%
% %% manually find edges
% disp('differentiating filters')
% diffx=[1 -1]
% diffy = diffx'
%
% %smoothing filters Previtt
% sx=ones(2,3);
% sy=sx';
%
% % build Previtt derivative filters
% disp('derivative filters Previtt')
% dx=conv2(sy , diffx);
% dy=conv2(sx , diffy);
%
% %smoothing filters Sobel
% sx=[1 2 1 ; 1 2 1];
% sy=sx';
%
% % Build Sobel derivative filters
% disp(' derivative filters Sobel')
% dx=conv2(sy,diffx);
% dy=conv2(sx,diffy);
%
% % lap = [0, 1,0; 1, -4, 0; 0,1,0];
% % im_2ord = uint8( conv2(im_g, lap));
% % im_2ord = im_2ord(2:end-1, 2:end-1);
% % figure; imshow(im_2ord);
% % k = 5;
% % figure; imshow(im_g +k*im_2ord);
%
%
% % gammaVal = 0.25;
% % im_d = im2double(im_g);
% % figure(4)
% % subplot(1,2,1)
% % imshow(im_d, [])
% % title('original image')
% % subplot(1,2,2)
% % imshow(im_d.^gammaVal, [])
% % title(sprintf('Gamma corrected image using \\gamma = %.2f', gammaVal));
%
% % try to enhance changes
%
%
% %%
% % compute gradient components (horizontal and vertical derivatives)
% Gx=conv2(im_g , dx , 'same');
% Gy=conv2(im_g , dy , 'same');
%
% figure; imshow(Gx, []),title('horizontal derivative')
% figure; imshow(Gy, []),title('vertical derivative')
%
% % Gradient Norm
% M=sqrt(Gx.^2 + Gy.^2);
%
% figure; imshow(M,[]),title('gradient magnitude')

% FNT_SZ = 20;
% hold on;
% [x y]=getpts
% plot(x,y,'.w','MarkerSize',12, 'LineWidth', 3); % plots points clicked by user with red circles
% a=[x(1) y(1) 1]';
% text(a(1), a(2), 'a', 'FontSize', FNT_SZ, 'Color', 'w')
% b=[x(2) y(2) 1]';
% text(b(1), b(2), 'b', 'FontSize', FNT_SZ, 'Color', 'w')
% c=[x(3) y(3) 1]';
% text(c(1), c(2), 'd', 'FontSize', FNT_SZ, 'Color', 'w')
% d=[x(4) y(4) 1]';
% text(d(1), d(2), 'c', 'FontSize', FNT_SZ, 'Color', 'w')
% %
% % e=[x(5) y(5) 1]';
% % text(e(1), e(2), 'e', 'FontSize', FNT_SZ, 'Color', 'w')
% % f=[x(6) y(6) 1]';
% % text(f(1), f(2), 'f', 'FontSize', FNT_SZ, 'Color', 'w')
% % g=[x(7) y(7) 1]';
% % text(g(1), g(2), 'g', 'FontSize', FNT_SZ, 'Color', 'w')
% % h=[x(8) y(8) 1]';
% % text(h(1), h(2), 'h', 'FontSize', FNT_SZ, 'Color', 'w')
%
% lab = cross(a, b);
% lbc = cross(b, c);
% lcd = cross(c, d);
% lda = cross(d, a);
%
% v1 = cross(lab, lcd);
% %
% % lef = cross(e, f);
% % lgh = cross(g, h);
% v2 = cross(lbc, lda);
%
% % remember these have to be normalized before plotting them
% v1 = v1/v1(3);
% v2 = v2/v2(3);
%
% horz = cross(v1, v2);
%
% plot([a(1), v1( 1)], [a(2), v1(2)], 'b');
% plot([d(1), v1(1)], [d(2), v1(2)], 'b');
% plot([b(1), v1(1)], [b(2), v1(2)], 'b');
% plot([c(1), v1(1)], [c(2), v1(2)], 'b');
%
% plot([a(1), v2(1)], [a(2), v2(2)], 'b');
% plot([c(1), v2(1)], [c(2), v2(2)], 'b');
% plot([b(1), v2(1)], [b(2), v2(2)], 'b');
% plot([d(1), v2(1)], [d(2), v2(2)], 'b');
%
% plot([v1(1), v2(1)], [v1(2), v2(2)], 'r--')
% text(v1(1), v1(2), 'v1', 'FontSize', FNT_SZ, 'Color', 'r')
% text(v2(1), v2(2), 'v2', 'FontSize', FNT_SZ, 'Color', 'r')
%
% hold off
%
% horz= horz/horz(3);
%
% % Hp = affine2d([1,0,0; 0,1,0; horz']);
% % figure, imshow( imwarp(imtransform(im,Hp)));
%
% X = [a, b, c, d];
% aP = [1; 1; 1];
% bP = [1; 500; 1];
% cP = [500; 500; 1];
% dP = [500; 1; 1];
% XP = [aP, bP, cP, dP];
%
% % estimate homography using DTL algorithm
% H = homographyEstimation(X, XP);
%
% Apply the trasformation to the image mapping pixel centers using H^-1 and bilinear interpolation
% J = imwarpLinear(im_g, H, [-size(im_g, 2)/2, -size(im_g, 1)/2, size(im_g, 2)/2, size(im_g, 1)/2]);
%
% figure, imagesc(J), colormap gray, axis equal;
%
% j = uint8(J);
% imshow(j);
%
% horz = horz/horz(3);
% Har = [1,0,0; 0,1,0; horz'];
% J = imwarpLinear(im_g, Har, [-size(im_g, 2)/2, -size(im_g, 1)/2, size(im_g, 2)/2, size(im_g, 1)/2]);
%
% figure, imagesc(J), colormap gray, axis equal;
