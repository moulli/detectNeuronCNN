function [centers, mask] = getNeurons(layer, limit, halfsize_reg, halfsize_keep)
% Function that detects potential neurons from a layer resulting from
% lightsheet imaging. limit = 500, halfsize_reg=3, halfsize_keep=4 work best


    %% Parameters and outputs
    
    % Parameters
    tic
    if halfsize_reg < 1
        error('Half size for regression should be at least 1')
    end
    if halfsize_keep < halfsize_reg
        error('Half size for points to keep should be at least equal to half size for regression')
    end
    slayer = size(layer);
    print_info = 0;
    layer_noborder = layer;
    layer_noborder([1:halfsize_keep, slayer(1)-halfsize_keep:slayer(1)], :) = limit-1;
    layer_noborder(:, [1:halfsize_keep, slayer(2)-halfsize_keep:slayer(2)]) = limit-1;
    % Outputs
    centers = zeros(0, 2);
    mask_cell = cell(0, 1);
    
    
    %% Segmentation
    
    warning('off')
    while 1
        %% Get pixel with highest intensity
        [maxmax, maxind] = max(layer_noborder(:));
        if maxmax < limit
            break
        end
        [xpt, ypt] = ind2sub(slayer, maxind);
        
        %% Isolate a few points and fit with polynomial poly22
        [Y, X] = meshgrid(ypt-halfsize_reg:ypt+halfsize_reg, xpt-halfsize_reg:xpt+halfsize_reg);
        Z = layer(xpt-halfsize_reg:xpt+halfsize_reg, ypt-halfsize_reg:ypt+halfsize_reg);
        polyfit = fit([X(:), Y(:)], Z(:), 'poly22');
        
        %% Isolate more points and keep only those whose regression estimate
        % are higher than minimum actual value of these points
        [Y, X] = meshgrid(ypt-halfsize_keep:ypt+halfsize_keep, xpt-halfsize_keep:xpt+halfsize_keep);
        X = X(:); Y = Y(:);
        Z = layer(xpt-halfsize_keep:xpt+halfsize_keep, ypt-halfsize_keep:ypt+halfsize_keep);
        Zpoly = polyfit(X, Y);
        QUANTILE1 = 0.15;
        keep_points = (Zpoly > quantile(Z(:), QUANTILE1));
        Xkeep = X(keep_points); Ykeep = Y(keep_points);
        indkeep = sub2ind(slayer, Xkeep, Ykeep);
        
        %% Check the 9 center points are ok and if enough pixels
        pts_in_keep_points = zeros(2*halfsize_keep+1);
        pts_in_keep_points(halfsize_keep:halfsize_keep+2, halfsize_keep:halfsize_keep+2) = 1;
        pts_in_keep_points = pts_in_keep_points(:) .* keep_points;
        pixels_limit = (halfsize_keep+1)^2;
        if sum(pts_in_keep_points) == 9 && sum(keep_points) >= pixels_limit
            % Store center in centers
            centers = cat(1, centers, [xpt, ypt]);
            % Save values in mask
            mask_cell = [mask_cell; {indkeep}];
        end
        
        %% Delete pixels in image
        QUANTILE2 = QUANTILE1;
        keep_points = (Zpoly > quantile(Z(:), QUANTILE2));
        keep_points(((2*halfsize_keep+1)^2-1)/2 + 1) = true; % in case center is not taken
        Xkeep = X(keep_points); Ykeep = Y(keep_points);
        indkeep = sub2ind(slayer, Xkeep, Ykeep);
        layer(indkeep) = limit - 1;
        layer_noborder(indkeep) = limit - 1;
        
        %% Information:
        if mod(size(centers, 1), 100) == 0
            fprintf(repmat('\b', 1, print_info));
            print_info = fprintf('%.0f neurons already found in %.3f seconds \n', [size(centers, 1), toc]);
        end
    end
    
    %% Get mask from mask_cell
    mask = zeros(slayer);
    for i = length(mask_cell):-1:1
        mask(mask_cell{i}) = i;
    end


end