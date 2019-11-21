clear; close all; clc


%% Load mlayer

load('C:\Users\Hippolyte Moulle\Desktop\mlayer.mat', 'mlayer');


%% Build regressors

% Size parameter
halfsize = 2;
% Compute actual size
tsize = 2*halfsize + 1;
% Constant regressor
reg1 = ones(tsize);
reg1 = reg1(:);
% Gaussian regressor
[X, Y] = meshgrid(1:tsize);
comat = 3 * eye(2);
reg2 = mvnpdf([X(:), Y(:)], [halfsize+1, halfsize+1], comat);
reg2 = reg2 / mean(reg2);

% Regressors final
reg = [reg1, reg2];

% Adding more regressors
comat = 3 * eye(2);
temp = mvnpdf([X(:), Y(:)], [halfsize+1, halfsize+1], comat);
reg = [reg, temp/mean(temp)];
comat = 3 * [1, 0.6; 0.6, 1];
temp = mvnpdf([X(:), Y(:)], [halfsize+1, halfsize+1], comat);
reg = [reg, temp/mean(temp)];
comat = 3 * [1, -0.6; -0.6, 1];
temp = mvnpdf([X(:), Y(:)], [halfsize+1, halfsize+1], comat);
reg = [reg, temp/mean(temp)];
comat = 4 * eye(2);
temp = mvnpdf([X(:), Y(:)], [halfsize+1, halfsize+1], comat);
reg = [reg, temp/mean(temp)];
comat = 4 * [1, 0.6; 0.6, 1];
temp = mvnpdf([X(:), Y(:)], [halfsize+1, halfsize+1], comat);
reg = [reg, temp/mean(temp)];
comat = 4 * [1, -0.6; -0.6, 1];
temp = mvnpdf([X(:), Y(:)], [halfsize+1, halfsize+1], comat);
reg = [reg, temp/mean(temp)];
comat = 5 * eye(2);
temp = mvnpdf([X(:), Y(:)], [halfsize+1, halfsize+1], comat);
reg = [reg, temp/mean(temp)];
comat = 5 * [1, 0.6; 0.6, 1];
temp = mvnpdf([X(:), Y(:)], [halfsize+1, halfsize+1], comat);
reg = [reg, temp/mean(temp)];
comat = 5 * [1, -0.6; -0.6, 1];
temp = mvnpdf([X(:), Y(:)], [halfsize+1, halfsize+1], comat);
reg = [reg, temp/mean(temp)];



% %% Filter mlayer
% 
% % Convert mlayer to doubles
% mlayer = double(mlayer);
% 
% % Convolution kernel bigger than gaussian regressor
% convkern = ones(tsize);
% 
% % Convolute and give right size
% mlayer = conv2(mlayer, convkern) / numel(convkern);
% mlayer = mlayer((((tsize+3)/2)+1):(size(mlayer, 1)-((tsize+3)/2)), (((tsize+3)/2)+1):(size(mlayer, 2)-((tsize+3)/2)));



%% Compute F-statistic for all points above limit using regressors

% Define limit
limit = 450;
% Convert mlayer to doubles
mlayer = double(mlayer);
% Define F-statistic and coefficients matrix
Fmat = zeros(size(mlayer));
coef1mat = zeros(size(mlayer));
coef2mat = zeros(size(mlayer));
% Launch algorithm
tic
printinfo = 0;
for ix = (halfsize+1):(size(mlayer, 1)-halfsize)
    for iy = (halfsize+1):(size(mlayer, 2)-halfsize)
        % Regress if above layer
        if mlayer(ix, iy) >= limit
            layer_zone = mlayer(ix-halfsize:ix+halfsize, iy-halfsize:iy+halfsize);
            [coeffs, ~, residuals, ~, stats] = regress(layer_zone(:), reg);
            Fmat(ix, iy) = stats(2);
            coef1mat(ix, iy) = coeffs(1);
            coef2mat(ix, iy) = coeffs(2);
        end
        % Provide information
        numpix = (ix-1)*(size(mlayer, 2)-2*halfsize) + iy;
        if mod(numpix, 10000) == 0
            fprintf(repmat('\b', 1, printinfo));
            printinfo = fprintf('%.0f pixels already done out of %.0f in %.3f seconds \n', [numpix, numel(Fmat), toc]);
        end
    end
end


%% Ok this is good, but now we need to find neurons' centers

% Some of the high F-statistics are due to a negative coefficient linking
% gaussian to layer. These are not neurons, but rather gaps between
% neurons, and need to be deleted
Fmat_corrected = Fmat .* (coef2mat > 0); 
% Fmat_corrected(Fmat_corrected < 4) = 0;

% % We need to get rid of low F-statistics, otherwise glow can be
% % misinterpretated as a neuron. In order to do that, we need to
% % differenciate neurons among many others, and isolated neurons in glow.
% % 2 parameters: limit for coef1mat and minimum for F-statistic in glow
% glow_limit = 600;
% min_Fstat_glow = 5;
% % Deduce pixels to keep
% pixels_studied = find(Fmat_corrected ~= 0);
% Fmat_corrected(coef1mat < glow_limit & Fmat_corrected < 4) = 0;
% mtemp = Fmat_corrected .* coef1mat;
% Fmat_corrected(mtemp < 3000) = 0;

% Then convolving with the gaussian regressor as kernel to smoothen 
gkernel = reshape(reg2, tsize, tsize);

halfsize_gkern = 4;
% Compute actual size
tsize_gkern = 2*halfsize_gkern + 1;
% Gaussian regressor
[X, Y] = meshgrid(1:tsize_gkern);
gkernel = mvnpdf([X(:), Y(:)], [halfsize_gkern+1, halfsize_gkern+1], eye(2));
gkernel = gkernel / mean(gkernel);
gkernel = reshape(gkernel, tsize_gkern, tsize_gkern);

gkernel = reshape(reg2, tsize, tsize);
halfsize_gkern = halfsize;
Fmat_conv = conv2(Fmat_corrected, gkernel);
sFconv = size(Fmat_conv);
Fmat_conv = Fmat_conv(halfsize_gkern+1:sFconv(1)-halfsize_gkern, halfsize_gkern+1:sFconv(2)-halfsize_gkern);
% Getting rid of point that are too low, ie that were just one or a couple
% of pixels with a high F-stat, instead of a whole cluster
Fmat_conv(Fmat_conv < sum(reg2)) = 0;

% Now taking as neurons' centers the points higher than all their neighbours
Cmat = zeros(0, 2);
for ix = (halfsize+1):(size(mlayer, 1)-halfsize)
    for iy = (halfsize+1):(size(mlayer, 2)-halfsize)
        % Compare points to its neighbours
        pt = Fmat_conv(ix, iy);
        neighbours = Fmat_conv(ix-1:ix+1, iy-1:iy+1);
        neighbours = neighbours(:);
        neighbours(5) = []; % delete points so that it's a strict inequality
        comparison = (pt > neighbours); % otherwise all points in background are selected
        % Keep point only if superior to all neighbours
        if all(comparison)
            Cmat = cat(1, Cmat, [ix, iy]);
        end
    end
end
        



