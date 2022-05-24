function[masks] = initialise(metric, radius, alpha, options)
%
% Author:      Stephanie Reynolds
% Date:        25/09/2017
% Supervisors: Pier Luigi Dragotti, Simon R Schultz
% Overview:    This function is used as the initialisation for the segmentation
%              algorithm. Peaks in the 2D summary image(s) are identified 
%              as candidate ROIs. Peaks are found with an built-in MATLAB 
%              image processing function 'imextendedmax'. Peaks with 
%              relative height (with respect to their neighbours) that are 
%              less than alpha x sigma are suppressed. Here, sigma is
%              the standard deviation of the summary image and alpha is a 
%              tuning parameter. 
% Reference:   Reynolds et al. (2016) ABLE: an activity-based level set 
%              segmentation algorithm for two-photon calcium imaging data
%
%
%%%%%%%%%%%%%%%   INPUTS    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% metric                     MxN summary image of video, usually the
%                            pixelwise cross-correlation (see crossCorr.m)
% radius                     radius of a cell
% alpha                      tuning parameter, peaks below alpha*sigma will be
%                            suppressed
% options                    A variable of type struct. In the following we
%                            describe the fields of this struct. 
% options.blur_radius        [Default: 1] If present, this is the radius of
%                            blurring applied to the summary images. 
%                            blurred with radius options.blur_radius.
% options.secondary_metric   M x N array corresponding to a summary image,
%                            e.g. the mean image. If this field is present, 
%                            initialisation is  performed on both the first 
%                            argument 'metric' and a second summary image. The value
%                            options.second_alpha (the value of alpha for the
%                            second summary image) must also be present. 
%
%%%%%%%%%%%%%%%   OUTPUTS   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% masks                      MxNxK array, where K is the number of ROIs found.
%                            In each sheet (masks(:,:,ii)): -1 indicates that a
%                            pixel is inside the i-th ROI, +1 indicates that the
%                            pixel is outside the i-th ROI.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dim     = size(metric);
maxSize = round(pi * radius^2 * 1.5);

% Blur the input image (so that only peaks wider than 1 pixel are detected)
if isfield(options, 'blur_radius')
    blur_radius =  options.blur_radius;
else 
    blur_radius = 1;
end

metric          = imgaussfilt(metric, blur_radius);

% Find peaks of metric
h                     = sqrt(nanvar(metric(:))); %before was nanvar
%   assignin('base', 'h', h)

metric(isnan(metric)) = 0;
% assignin('base', 'metric', metric)


BW                    = imextendedmax(metric, alpha*h);
% assignin('base', 'matrix_alphatimesh', alpha*h)

% assignin('base', 'BW', BW)


% Each 4-connected component is an ROI
CC         = bwconncomp(BW, 4);
% assignin('base', 'CC', CC)

obj_num    = CC.NumObjects;
% assignin('base', 'obj_num', obj_num)

pixels     = CC.PixelIdxList;
% assignin('base', 'pixels', pixels)

masks      = zeros(dim(1), dim(2), obj_num);
%  assignin('base', 'masks', masks)

for ii  = 1:CC.NumObjects
    mask             = zeros(dim);
    mask(pixels{ii}) = 1;
    masks(:,:,ii)    = mask;
end
%  assignin('base', 'masks', masks)

% Remove any that are too large 
nnzs                          = squeeze(sum(sum(masks,1),2));
masks(:,:,nnzs > maxSize)     = [];

% Final ROIs
masks                         = -1*masks + ~masks;
% assignin('base', 'masks', masks)


end









