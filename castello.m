im_name = 'Image - Castello di Miramare.jpg';
im=imread(im_name);
whos im


imshow(im);

hold on;
[x y]=getpts
plot(x,y,'.w','MarkerSize',12, 'LineWidth', 3); % plots points clicked by user with red circles
a=[x(1) y(1) 1]';
text(a(1), a(2), 'a', 'FontSize', FNT_SZ, 'Color', 'w')
b=[x(2) y(2) 1]';
text(b(1), b(2), 'b', 'FontSize', FNT_SZ, 'Color', 'w')
c=[x(3) y(3) 1]';
text(c(1), c(2), 'c', 'FontSize', FNT_SZ, 'Color', 'w')
e=[x(4) y(4) 1]';
text(e(1), e(2), 'e', 'FontSize', FNT_SZ, 'Color', 'w')
