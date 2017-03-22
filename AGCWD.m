
%  Efficient Contrast Enhancement Using Adaptive Gamma Correction With Weighting Distribution
function im_out = AGCWD(img)

[row,col,~] = size(img);
% --- RGB 2 HSV ---
HSV = rgb2hsv(img/255);
V = HSV(:,:,3);

[counts, x] = imhist(V);
pdf = counts/(sum(counts));

% alph: adjusted parameter
alph = 0.8;
% --- WD -----
pdf_max = max(pdf);
pdf_min = min(pdf);

pdf_w = pdf_max*((pdf - pdf_min)./(pdf_max - pdf_min)).^alph; 

% --- weighted cdf -----
% lmax: maximum intensity of input
lmax_idx = (find(counts, 1, 'last'));
lmax = max(V(:));
sum_pdf_w = 0;
all_pdf_w = sum(pdf_w);
for i=1:lmax_idx
    sum_pdf_w = sum_pdf_w + pdf_w(i);
    cdf_w(i) = sum_pdf_w./all_pdf_w;
end

gamma =1-cdf_w;
% ---- Enhancement ----
V = reshape(V,row*col,1);
T = zeros(size(V));
for i=1:lmax_idx
    L = V(V==x(i));
    T(V==x(i)) = lmax*(L./lmax).^gamma(i);
end
V2 = reshape(T,row,col);
HSV(:,:,3) = V2;
im_out = hsv2rgb(HSV);
% im_out = uint8(im_out*255);

