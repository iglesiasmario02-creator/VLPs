clear all
close all

% Load the image
image = imread('day1/beads/Image001.tif');

% Get the dimensions of the image
[rows, cols, channels] = size(image);

% Check and pad the image if it is 1199 x 512
if cols == 1199 && rows == 512
    padded_image = zeros(1200, 512, channels, 'like', image);
    padded_image(1:1199, :, :) = image;
    image = padded_image;
    cols = 1200;
end

% Verify the dimensions
if cols ~= 1200 || rows ~= 512
    error('The image size must be 512 x 1200 or 512 x 1199');
end

% Split the image into two 600 x 512 images
image1 = image(:, 1:600, :);
image2 = image(:, 601:end, :);

% Convert images to grayscale if they are RGB
if channels == 3
    image1 = rgb2gray(image1);
    image2 = rgb2gray(image2);
end

% Normalize the images using imadjust
image1 = imadjust(image1);
image2 = imadjust(image2);

% % % check that it works
%  image2 = imtranslate(image2, [5, 5]);

% Detect SURF features
points1 = detectSURFFeatures(image1);
points2 = detectSURFFeatures(image2);

% Extract features
[features1, valid_points1] = extractFeatures(image1, points1);
[features2, valid_points2] = extractFeatures(image2, points2);

% Match features
indexPairs = matchFeatures(features1, features2);

matchedPoints1 = valid_points1(indexPairs(:, 1));
matchedPoints2 = valid_points2(indexPairs(:, 2));

% Estimate the transformation matrix
tform = estimateGeometricTransform(matchedPoints2, matchedPoints1, 'affine');

% Apply the transformation
image2_transformed = imwarp(image2, tform, 'OutputView', imref2d(size(image1)));

% Overlay the images
combined_image_transformed = cat(3, image1, image2_transformed, zeros(size(image1)));
combined_image_untransformed = cat(3, image1, image2, zeros(size(image1)));

% Display the images with features
figure;
subplot(2, 2, 1);
imshow(image1);
title('Image 1');
hold on;
plot(valid_points1.selectStrongest(50));
hold off;

subplot(2, 2, 2);
imshow(image2);
title('Image 2');
hold on;
plot(valid_points2.selectStrongest(50));
hold off;

subplot(2, 2, 3);
imshow(combined_image_transformed);
title('Transformed Image 2 on Image 1 (RGB)');

subplot(2, 2, 4);
imshow(combined_image_untransformed);
title('Overlay of Untransformed Images (RGB)');

% Link the axes of the four images
linkaxes(findall(gcf, 'Type', 'axes'), 'xy');

% Save the transformation
save('transformation.mat', 'tform');

%%

% Load a new image to check transformation stability
new_image = imread('day1/beads/Image034.tif');

% Split the new image into two 600 x 512 images
new_image1 = new_image(:, 1:600, :);
new_image2 = new_image(:, 601:end, :);

% Convert new images to grayscale if they are RGB
if size(new_image, 3) == 3
    new_image1 = rgb2gray(new_image1);
    new_image2 = rgb2gray(new_image2);
end

% Normalize the new images using imadjust
new_image1 = imadjust(new_image1);
new_image2 = imadjust(new_image2);

% Apply the saved transformation to the new image2
new_image2_transformed = imwarp(new_image2, tform, 'OutputView', imref2d(size(new_image1)));

% Overlay the images
combined_image_transformed = cat(3, new_image1, new_image2_transformed, zeros(size(new_image1)));
combined_image_untransformed = cat(3, new_image1, new_image2, zeros(size(new_image1)));

% Display the images with matched features
figure;
subplot(2, 2, 1);
imshow(new_image1);
title('New Image 1');

subplot(2, 2, 2);
imshow(new_image2);
title('New Image 2');

subplot(2, 2, 3);
imshow(combined_image_transformed);
title('Transformed New Image 2 on New Image 1 (RGB)');

subplot(2, 2, 4);
imshow(combined_image_untransformed);
title('Overlay of Untransformed New Images (RGB)');

% Link the axes of the four images
linkaxes(findall(gcf, 'Type', 'axes'), 'xy');