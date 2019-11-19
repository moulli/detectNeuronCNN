function saveNeurons(layer, centers, mask_contour)
% Function that saves potential neurons from a layer resulting from
% lightsheet imaging. centers and mask are the outputs of getNeurons.m.
% Files are saved as a .mat and as a .png


    %% Main loop
    
    for i = 1:length(mask_contour)
        % Plot image with n-pixel bordel
        h = figure('Menu', 'none', 'ToolBar', 'none'); 
        ah = axes('Units', 'Normalized', 'Position', [0, 0, 1, 1]);
        set(h,'Visible','off');
        hold on
        co = centers(i, :);
        n = 20; 
        if co(1)-n <= 0 || co(1)+n > size(layer, 1) || co(2)-n <= 0 || co(2)+n > size(layer, 2)
            continue
        end
        pixels = layer(co(1)-n:co(1)+n, co(2)-n:co(2)+n);
        image(pixels, 'CDataMapping', 'scaled')
        % Plot center of image and localization circle
        scatter(n+1, n+1, '.k')
%         x = n+1; y = n+1; r = 5;
%         ang=0:0.01:2*pi; 
%         xp=r*cos(ang);
%         yp=r*sin(ang);
%         plot(x+xp,y+yp, 'k');
        plot(mask_contour{i}(1, :), mask_contour{i}(2, :), 'k')
        axis equal 
        axis off        
        % Define name
        namelen = 9;
        namenum = num2str(round((10^namelen)*rand));
        namenum = strcat(repmat('0', 1, namelen-length(namenum)), namenum);
        % Save .mat file
        matpath = fullfile('images_to_label', 'mat', strcat(namenum, '.mat'));
        save(matpath, 'pixels');
        % Save .png file
        pngpath = fullfile('images_to_label', 'png', strcat(namenum, '.png'));
        img = getframe(gca);
        pngpad = img.cdata;
        padding = (size(pngpad, 2)-size(pngpad, 1))/2;
        pngfile = pngpad(:, padding+1:size(pngpad, 2)-padding, :);
        imwrite(pngfile, pngpath);
    %     saveas(h, pngpath, 'png');
    %     hgexport(h, jpgpath, hgexport('factorystyle'), 'Format', 'png');
    end

end