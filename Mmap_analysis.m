clear; close all; clc


%% Focus and create Mmap:

addpath(genpath('/home/ljp/Programs'));

root = '/home/ljp/Science/GeoffreysComputer/Projects/RLS/';
study = '';
date = '2019-05-16';
run = 01;
F = NT.Focus(root, study, date, run);

% m = Mmap(F.tag('corrected'));
m = adapted4DMatrix(F, 'corrected');


%% Loading one layer on maximizing:

layer = 12;
mlayer = permute(m(:, :, 10, :), [1, 2, 4, 3]);
mlayer = max(mlayer, [], 3);


%% Algorithm to detect neurons:

% % Resizing mlayer:
% mlayertemp = double(mlayer(201:300, 601:700));
% slayer = size(mlayertemp);
% % Main algorithm:
% mdiffx = zeros(slayer-2);
% mdiffy = zeros(slayer-2);
% mdifftopx = zeros(slayer-2);
% mdifftopy = zeros(slayer-2);
% for i = 2:(slayer(1)-1)
%     mdiffx(:, i-1) = (mlayertemp(2:end-1, i+1)-mlayertemp(2:end-1, i) > 0) & (mlayertemp(2:end-1, i)-mlayertemp(2:end-1, i-1) > 0);
%     mdiffy(i-1, :) = (mlayertemp(i+1, 2:end-1)-mlayertemp(i, 2:end-1) > 0) & (mlayertemp(i, 2:end-1)-mlayertemp(i-1, 2:end-1) > 0);
%     mdifftopx(:, i-1) = (mlayertemp(2:end-1, i+1)-mlayertemp(2:end-1, i) > 0) & (mlayertemp(2:end-1, i)-mlayertemp(2:end-1, i-1) < 0);
%     mdifftopy(i-1, :) = (mlayertemp(i+1, 2:end-1)-mlayertemp(i, 2:end-1) > 0) & (mlayertemp(i, 2:end-1)-mlayertemp(i-1, 2:end-1) < 0);
% end


%% Algorithm discovering map:

% figure('units', 'normalized', 'outerposition', [0 0 1 1])
% subplot(1, 2, 1)
% image(mlayertemp, 'CDataMapping', 'scaled')
% colorbar
% 
% mlayer_ind = ones(slayer);
% mneurons = zeros([slayer, 0]);
% while ~isequal(mlayer_ind, zeros(slayer))
%     
%     % Chosing which action to do:
%     nleft = find(mlayer_ind == -1);
%     if isempty(nleft)
%         % Adding new neuron:
%         mat_temp = mlayer_ind .* mlayertemp;
%         [~, next_pt] = max(mat_temp(:));
%         [xpt, ypt] = ind2sub(slayer, next_pt);
%         mneurons = cat(3, mneurons, zeros(slayer));
%     else
%         % Adding waiting neuron:
%         [xpt, ypt] = ind2sub(slayer, nleft(1));
%     end
%     % Changing parameters:
%     mneurons(xpt, ypt, end) = 1;
%     % Setting indication to 0:
%     mlayer_ind(xpt, ypt) = 0;
%     
%     % Adding waiting points:
%     % In x-axis direction:
%     if xpt == 1
%         neix = xpt + 1; 
%     elseif xpt == slayer(1)
%         neix = xpt - 1;
%     else 
%         neix = [xpt-1; xpt+1];
%     end
%     % In y-axis direction:
%     if ypt == 1
%         neiy = ypt + 1;
%     elseif ypt == slayer(2)
%         neiy = ypt - 1;
%     else
%         neiy = [ypt-1; ypt+1];
%     end
%     % Forming final couples:
%     nex = zeros(length(neix)*length(neiy), 1);
%     ney = zeros(length(neix)*length(neiy), 1);
%     for i = 1:length(neix)
%         for j = 1:length(neiy)
%             nex(length(neiy)*(i-1)+j) = neix(i);
%             ney(length(neiy)*(i-1)+j) = neiy(j);
%         end
%     end
%     % Analyzing each point independently:
%     for k = 1:(length(neix)*length(neiy))
%         if mlayer_ind(nex(k), ney(k)) ~= 0 && mlayertemp(nex(k), ney(k)) < mlayertemp(xpt, ypt)
%             mlayer_ind(nex(k), ney(k)) = -1;
%         end
%     end
%     
% %     % Plotting:
% %     pause(0.05)
% %     subplot(1, 2, 2)
% %     image(mlayertemp .* 2 .* mlayer_ind(:, :, end), 'CDataMapping', 'scaled')
% end


%% Works really badly, let's try with gaussians:

LIMIT = 450;
halfsize_neuron = 2;
mlayer_copy = mlayer;
mlayer = double(mlayer);
mlayer_gauss = double(mlayer);
figure; image(mlayer_gauss, 'CDataMapping', 'scaled'); colorbar; axis equal
slayer = size(mlayer_gauss);
mlayer_ind = ones(slayer);
centers_layer = zeros(0, 2);
DEG = zeros(0, 4);
% mneurons = zeros([slayer, 0]);
mneurons = {};
[Ylay, Xlay] = meshgrid(1:slayer(2), 1:slayer(1));
Clay = [Xlay(:), Ylay(:)];
print_info = 0;
tic
while 1
    [maxmax, maxind] = max(mlayer_gauss(:));
    if maxmax < LIMIT
        break
    end
    [xpt, ypt] = ind2sub(slayer, maxind);
    % Find differences in layers
    deg_lay1 = [mlayer(xpt-1, ypt), mlayer(xpt+1, ypt), mlayer(xpt, ypt-1), mlayer(xpt, ypt+1)];
    deg_lay2 = [mlayer(xpt-2, ypt), mlayer(xpt+2, ypt), mlayer(xpt, ypt-2), mlayer(xpt, ypt+2)];
    deg_lay = [maxmax-deg_lay1]; %, deg_lay1-deg_lay2];
    
    
    pts = zeros(0, 2);
    for k1 = -halfsize_neuron:halfsize_neuron
        for k2 = -halfsize_neuron:halfsize_neuron
            xtemp = max(1, xpt+k1); xtemp = min(slayer(1), xtemp);
            ytemp = max(1, ypt+k2); ytemp = min(slayer(2), ytemp);
            pts_temp = repmat([xtemp, ytemp], round(mlayer(xtemp, ytemp)), 1);
            pts = [pts; pts_temp];
        end
    end
    GMMtemp = fitgmdist(pts, 1);
    mu = GMMtemp.mu;
    sigma = GMMtemp.Sigma;
    A = maxmax * sqrt((2*pi)^2*det(sigma));
    % Computing gaussian:
    gauss_temp = A * mvnpdf(unique(pts, 'rows'), mu, sigma);
    minus_gauss = reshape(A * mvnpdf(Clay, mu, sigma), slayer);
    mlayer_gauss = mlayer_gauss - minus_gauss;
    % Adding neuron:
    minus_gauss(minus_gauss < 0.001) = 0;
%     mneurons = cat(3, mneurons, minus_gauss);
    if all(deg_lay >= -5)
        DEG = cat(1, DEG, deg_lay);
        centers_layer = cat(1, centers_layer, [xpt, ypt]);
        mneurons = [mneurons; {sparse(minus_gauss)}];
    end
    % Information:
    if mod(length(mneurons), 100) == 0
        fprintf(repmat('\b', 1, print_info));
        print_info = fprintf('%.0f neurons already found in %.3f seconds \n', [length(mneurons), toc]);
    end
end
temp = zeros(slayer);
for i = 1:length(mneurons); temp = temp + full(mneurons{i}); end
figure; image(temp, 'CDataMapping', 'scaled'); colorbar; axis equal
path_created = fullfile('/home/ljp/Science/Hippolyte', 'mneurons.mat');
save(path_created, 'mneurons');

% Plotting centers of neurons
figure
hold on
image(mlayer, 'CDataMapping', 'scaled')
colorbar
axis equal
plot(centers_layer(:, 2), centers_layer(:, 1), '.r')

    
    
    
    
    
    
    
    