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


%% Compute F-statistic for all points above limit using regressors

% Define limit
limit = 0;
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
            [coeffs, ~, ~, ~, stats] = regress(layer_zone(:), reg);
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


%% Cleaning the Fmat result

% Delete values for coef2mat under a certain value
Fcorrected = Fmat .* (coef2mat > 12);

% Convolve with reg2
gkernel = reshape(reg2, tsize, tsize) - min(reg2);
% gkernel([1, end], :) = 0;
% gkernel(:, [1, end]) = 0;
Fconv = conv2(Fcorrected, gkernel);
sFconv = size(Fconv);
Fconv = Fconv(halfsize+1:sFconv(1)-halfsize, halfsize+1:sFconv(2)-halfsize);


%% Compute coefficient 2 for all points in Fconv

coef2mat_N = zeros(size(mlayer));
% Launch algorithm
tic
printinfo = 0;
for ix = (halfsize+1):(size(mlayer, 1)-halfsize)
    for iy = (halfsize+1):(size(mlayer, 2)-halfsize)
        layer_zone = Fconv(ix-halfsize:ix+halfsize, iy-halfsize:iy+halfsize);
        coeffs = regress(layer_zone(:), reg);
        coef2mat_N(ix, iy) = coeffs(2);
        % Provide information
        numpix = (ix-1)*(size(mlayer, 2)-2*halfsize) + iy;
        if mod(numpix, 10000) == 0
            fprintf(repmat('\b', 1, printinfo));
            printinfo = fprintf('%.0f pixels already done out of %.0f in %.3f seconds \n', [numpix, numel(Fmat), toc]);
        end
    end
end


%% Now taking as neurons' centers the points higher than all their neighbours in coef2mat_N

Cmat = zeros(0, 2);
for ix = (halfsize+1):(size(mlayer, 1)-halfsize)
    for iy = (halfsize+1):(size(mlayer, 2)-halfsize)
        % Compare points to its neighbours
        pt = coef2mat_N(ix, iy);
        neighbours = coef2mat_N(ix-1:ix+1, iy-1:iy+1);
        neighbours = neighbours(:);
        neighbours(5) = []; % delete points so that it's a strict inequality
        comparison = (pt > neighbours); % otherwise all points in background are selected
        % Keep point only if superior to all neighbours
        if all(comparison)
            Cmat = cat(1, Cmat, [ix, iy]);
        end
    end
end


%% Plotting the different information

figure
image(Fmat, 'CDataMapping', 'scaled')
colorbar
axis equal

figure
image(Fcorrected, 'CDataMapping', 'scaled')
colorbar
axis equal

figure
image(Fconv, 'CDataMapping', 'scaled')
colorbar
axis equal

figure
image(coef2mat_N, 'CDataMapping', 'scaled')
colorbar
axis equal

figure
hold on
image(mlayer, 'CDataMapping', 'scaled')
colorbar
plot(Cmat(:, 2), Cmat(:, 1), '.r')
axis equal

