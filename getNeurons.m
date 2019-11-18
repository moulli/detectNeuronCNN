function [centers, mask] = getNeurons(layer, limit, halfsize_reg, halfsize_keep)
% Function that detects potential neurons from a layer resulting from
% lightsheet imaging. halfsize_reg=3, halfsize_keep=4

    %% Parameters and outputs
    
    % Parameters
    tic
    slayer = size(layer);
    print_info = 0;
    % Outputs
    centers = zeros(0, 2);
    mask = zeros(size(layer));
    
    
    %%
    
    warning('off')
    neuron_number = 1;
    while 1
        % Get pixel with highest intensity
        [maxmax, maxind] = max(layer(:));
        if maxmax < limit
            break
        end
        [xpt, ypt] = ind2sub(slayer, maxind);
        % Isolate a few points and fit with polynomial poly22
        [Y, X] = meshgrid(ypt-halfsize_reg:ypt+halfsize_reg, xpt-halfsize_reg:xpt+halfsize_reg);
        Z = layer(xpt-halfsize_reg:xpt+halfsize_reg, ypt-halfsize_reg:ypt+halfsize_reg);
        polyfit = fit([X(:), Y(:)], Z(:), 'poly22');
        % Isolate more points and keep only those whose regression estimate
        % are higher than minimum actual value of these points
        [Y, X] = meshgrid(ypt-halfsize_keep:ypt+halfsize_keep, xpt-halfsize_keep:xpt+halfsize_keep);
        X = X(:); Y = Y(:);
        Z = layer(xpt-halfsize_keep:xpt+halfsize_keep, ypt-halfsize_keep:ypt+halfsize_keep);
        Zpoly = polyfit(X, Y);
        keep_points = (Zpoly > quantile(Z(:), 0.25));
        keep_points(((2*halfsize_keep+1)^2-1)/2 + 1) = true; % in case center is not taken
        Xkeep = X(keep_points); Ykeep = Y(keep_points);
        indkeep = sub2ind(slayer, Xkeep, Ykeep);
        % Keep neuron only if enough pixels
        if sum(keep_points) >= ((2*halfsize_keep+1)^2-1)/2 + 1
            % Store center in centers
            centers = cat(1, centers, [xpt, ypt]);
            % Save values in mask
            mask(indkeep) = neuron_number;
            neuron_number = neuron_number + 1;
        end
        % Delete pixels in image
        keep_points = (Zpoly > quantile(Z(:), 0.15));
        keep_points(((2*halfsize_keep+1)^2-1)/2 + 1) = true; % in case center is not taken
        Xkeep = X(keep_points); Ykeep = Y(keep_points);
        indkeep = sub2ind(slayer, Xkeep, Ykeep);
        layer(indkeep) = limit - 1;
        % IT WOULD BE INTERESTING TO DELETE EACH PIXEL THAT IS OF LOWER
        % INTENSITY THAN ANY OF THOSE SELECTED. KIND OF GOING DOWN UNTIL IT
        % GOES UP. BUT WE WOULD MAKE ANOTHER LAYER AND TAKE THE HIGHEST
        % POINTS FROM THIS LAYER WITH LESS AND LESS PIXELS, WHEREAS THE
        % ACTUAL VALUES FOR REGRESSION WOULD BE TAKEN FROM NORMAL LAYER,
        % THE ONE USED UNTIL NOW. MAYBE NOT ACTUALLY
        
        % Information:
        if mod(size(centers, 1), 100) == 0
            fprintf(repmat('\b', 1, print_info));
            print_info = fprintf('%.0f neurons already found in %.3f seconds \n', [size(centers, 1), toc]);
        end
    end
    warning('on')


end