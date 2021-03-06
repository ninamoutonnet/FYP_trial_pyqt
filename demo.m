function [masks, cell_ts, nhbd_ts, corrIm, smaller_ROIs, larger_ROIs] = demo(name, radius, alpha, blur_radius, lambda_param, mergeCorr, metric, maxlt)

    tiff_info = imfinfo(name); % return tiff structure, one element per image
    tiff_stack = imread(name, 1) ; % read in first image
    %concatenate each successive tiff to tiff_stack
    for ii = 2 : size(tiff_info, 1)
        temp_tiff = imread(name, ii);
        tiff_stack = cat(3 , tiff_stack, temp_tiff);
    end
    video = tiff_stack;
    meanIm = mean(video,3);
    corrIm = crossCorr(video);

    %% Initialisation
    % Parameters
    init_opt.blur_radius        = double(blur_radius);
    phi_0          = initialise(corrIm, double(radius), double(alpha), init_opt);

    % Refresh workspace
    close all; clear seg_opt;

    % Algorithm parameters
    seg_opt.lambda              = double(lambda_param);
    seg_opt.mergeCorr           = double(mergeCorr);
    seg_opt.mergeDuring         = 1;
    seg_opt.metric              = metric;
    seg_opt.maxlt               = double(maxlt);

    % Do segmentation, segmenting about 300 ROIs takes ~15 mins
    tic;
    [masks, cell_ts, nhbd_ts] = segment(phi_0, video, double(radius), seg_opt);
    runtime = toc

    % Get number of pixels in each mask
    pix_num = zeros(size(masks,3),1);
    for mask_num = 1:size(masks,3)
       pix_num(mask_num) = nnz(masks(:,:,mask_num));
    end

    % Threshold:
    % If a user is only looking for cell bodies, it is beneficial to threshold
    % the size of the ROIs.
    min_size     = pi*radius^2*0.5;
    smaller_ROIs = masks(:,:, pix_num < min_size);
    larger_ROIs  = masks(:,:, pix_num >= min_size);


end
