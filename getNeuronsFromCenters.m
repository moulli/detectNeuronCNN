function mask_contour = getNeuronsFromCenters(mlayer, centers)
% This function returns mask contours based on neurons centers


    %% Get meshgrid of indexes from mlayer
    
    slayer = size(mlayer);
    [Xlay, Ylay] = meshgrid(1:slayer(1), 1:slayer(2));
    Xlay = Xlay(:);
    Ylay = Ylay(:);


    %% Loop over neurons centers
    
    mask_contour = cell(size(centers, 1), 1);
    for i = 1:size(centers, 1)
        
        %% Get neighbouring points from neuron's center and regress
        xpt = centers(i, 1);
        ypt = centers(i, 2);
        neighb = 1;
        [Y, X] = meshgrid(ypt-neighb:ypt+neighb, xpt-neighb:xpt+neighb);
        Z = mlayer(xpt-neighb:xpt+neighb, ypt-neighb:ypt+neighb);
        polyfit = fit([X(:), Y(:)], Z(:), 'poly22');    
        
        %% Compute regression for Xlay, Ylay and only keep positive points
        layReg = polyfit(Xlay, Ylay);
        Xkeep = Xlay(layReg > 0);
        Ykeep = Ylay(layReg > 0);
        indkeep = sub2ind(slayer, Xkeep, Ykeep);
        
        %% Now keep only regression points that are above a limit value
        limit = mean(mlayer(indkeep));
        Xkeep = Xlay(layReg > limit);
        Ykeep = Ylay(layReg > limit);
        indkeep = sub2ind(slayer, Xkeep, Ykeep);
        
        
        %% Now compute contour
        contour_i = zeros(slayer);
        contour_i(indkeep) = 1;
        contour_i = imcontour(contour_i, 1);
        mask_contour{i} = contour_i(:, contour_i(1, :)~=0.5);
        
    end    


end