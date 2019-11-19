function mask_contour = getMaskcontour(mask)
% Computes neurons contours from mask (get mask from getNeurons)

    tic
    slayer = size(mask);
    numneu = max(mask(:));
    print_info = fprintf('Computing mask contour \n');
    mask_contour = cell(numneu, 1);
    warning('off')
    for i = 1:numneu
        %% Add neuron's contour
        contour_i = zeros(slayer);
        contour_i(mask == i) = 1;
        contour_i = imcontour(contour_i, 1);
        mask_contour{i} = contour_i(:, contour_i(1, :)~=0.5);
        %% Information
        if mod(i, 100) == 0
            fprintf(repmat('\b', 1, print_info));
            print_info = fprintf('%.0f neurons contours already found in %.3f seconds \n', [i, toc]);
        end
    end
    warning('on')
    
end