clear; close all; clc


%% Focus and create Mmap:

addpath(genpath('/home/ljp/Programs'));

root = '/home/ljp/Science/GeoffreysComputer/Paper_Data/2018_Migault/';
study = '';
date = '2018-05-24';
run = 08;
F = NT.Focus(root, study, date, run);

m = Mmap(F.tag('corrected'));


%% Loading one layer on maximizing:

layer = 10;
mlayer = permute(m(:, :, 10, :), [1, 2, 4, 3]);
mlayer = max(mlayer, [], 3);


%% Algorithm to detect neurons:

% Resizing mlayer:
mlayertemp = double(mlayer(201:300, 601:700));
slayer = size(mlayertemp);
% Main algorithm:
mdiffx = zeros(slayer-2);
mdiffy = zeros(slayer-2);
mdifftopx = zeros(slayer-2);
mdifftopy = zeros(slayer-2);
for i = 2:(slayer(1)-1)
    mdiffx(:, i-1) = (mlayertemp(2:end-1, i+1)-mlayertemp(2:end-1, i) > 0) & (mlayertemp(2:end-1, i)-mlayertemp(2:end-1, i-1) > 0);
    mdiffy(i-1, :) = (mlayertemp(i+1, 2:end-1)-mlayertemp(i, 2:end-1) > 0) & (mlayertemp(i, 2:end-1)-mlayertemp(i-1, 2:end-1) > 0);
    mdifftopx(:, i-1) = (mlayertemp(2:end-1, i+1)-mlayertemp(2:end-1, i) > 0) & (mlayertemp(2:end-1, i)-mlayertemp(2:end-1, i-1) < 0);
    mdifftopy(i-1, :) = (mlayertemp(i+1, 2:end-1)-mlayertemp(i, 2:end-1) > 0) & (mlayertemp(i, 2:end-1)-mlayertemp(i-1, 2:end-1) < 0);
end


%% Algorithm discovering map:

figure('units', 'normalized', 'outerposition', [0 0 1 1])
subplot(1, 2, 1)
image(mlayertemp, 'CDataMapping', 'scaled')
colorbar

mlayer_ind = ones(slayer);
mneurons = zeros([slayer, 0]);
while ~isequal(mlayer_ind, zeros(slayer))
    
    % Chosing which action to do:
    nleft = find(mlayer_ind == -1);
    if isempty(nleft)
        % Adding new neuron:
        mat_temp = mlayer_ind .* mlayertemp;
        [~, next_pt] = max(mat_temp(:));
        [xpt, ypt] = ind2sub(slayer, next_pt);
        mneurons = cat(3, mneurons, zeros(slayer));
    else
        % Adding waiting neuron:
        [xpt, ypt] = ind2sub(slayer, nleft(1));
    end
    % Changing parameters:
    mneurons(xpt, ypt, end) = 1;
    % Setting indication to 0:
    mlayer_ind(xpt, ypt) = 0;
    
    % Adding waiting points:
    % In x-axis direction:
    if xpt == 1
        neix = xpt + 1; 
    elseif xpt == slayer(1)
        neix = xpt - 1;
    else 
        neix = [xpt-1; xpt+1];
    end
    % In y-axis direction:
    if ypt == 1
        neiy = ypt + 1;
    elseif ypt == slayer(2)
        neiy = ypt - 1;
    else
        neiy = [ypt-1; ypt+1];
    end
    % Forming final couples:
    nex = zeros(length(neix)*length(neiy), 1);
    ney = zeros(length(neix)*length(neiy), 1);
    for i = 1:length(neix)
        for j = 1:length(neiy)
            nex(length(neiy)*(i-1)+j) = neix(i);
            ney(length(neiy)*(i-1)+j) = neiy(j);
        end
    end
    % Analyzing each point independently:
    for k = 1:(length(neix)*length(neiy))
        if mlayer_ind(nex(k), ney(k)) ~= 0 && mlayertemp(nex(k), ney(k)) < mlayertemp(xpt, ypt)
            mlayer_ind(nex(k), ney(k)) = -1;
        end
    end
    
    % Plotting:
    pause(0.05)
    subplot(1, 2, 2)
    image(mlayertemp .* 2 .* mlayer_ind(:, :, end), 'CDataMapping', 'scaled')
end


%% Works really badly, let's try with gaussians:

mlayer_ind = ones(slayer);
mneurons = zeros([slayer, 10]);
for i = 1:10
    [maxmax, maxind] = max(mlayertemp(:));
    [xpt, ypt] = ind2sub(slayer, maxind);
    pts = zeros(0, 2);
    for k1 = -2:2
        for k2 = -2:2
            pts = [pts; xpt+k1, ypt+k2];
        end
    end
    GMMtemp = fitgmdist(pts, 1);
    mu = GMMtemp.mu;
    sigma = GMMtemp.Sigma;
    A = maxmax * sqrt((2*pi)^2*det(sigma));
    
end
    

    
    
    
    
    
    
    
    