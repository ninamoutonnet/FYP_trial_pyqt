function [phi_0, masks, cell_ts, nhbd_ts] = demo(name)

    disp('here1!')
    tiff_info = imfinfo(name); % return tiff structure, one element per image
    tiff_stack = imread(name, 1) ; % read in first image
    disp('here2!')
    %concatenate each successive tiff to tiff_stack
    for ii = 2 : size(tiff_info, 1)
        temp_tiff = imread(name, ii);
        tiff_stack = cat(3 , tiff_stack, temp_tiff);
    end
    disp('here3!')
    video = tiff_stack;
    meanIm = mean(video,3);
    corrIm = crossCorr(video);

    %% Initialisation
    % Parameters
    radius                      = 7;
    alpha                       = 0.55;
    init_opt.blur_radius        = 1.5;
    disp('here4!')
    phi_0          = initialise(corrIm, radius, alpha, init_opt);


    % Refresh workspace
    close all; clear seg_opt;

    % Algorithm parameters
    seg_opt.lambda              = 10;
    seg_opt.mergeCorr           = 0.95;
    seg_opt.mergeDuring         = 1;
    seg_opt.metric              = 'corr';
    seg_opt.maxlt               = 150;

    % Do segmentation, segmenting about 300 ROIs takes ~15 mins
    tic;
    [masks, cell_ts, nhbd_ts] = segment(phi_0, video, radius, seg_opt);
    runtime = toc

    % Get number of pixels in each mask
    pix_num = zeros(size(masks,3),1);
    for mask_num = 1:size(masks,3)
       pix_num(mask_num) = nnz(masks(:,:,mask_num));
    end


end