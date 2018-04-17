%% filtering
clc;clear all;close all;
% Create a cascade detector object.
faceDetector = vision.CascadeObjectDetector();

% Read a video frame and run the face detector.
video           = '2.mov';
videoFileReader = vision.VideoFileReader(video);
file            = VideoReader(video);
videoFrame      = step(videoFileReader);
bbox            = step(faceDetector, videoFrame);

% Finding the biggest box
bbox_size = size(bbox);
for i = 1:bbox_size(1)
    area(i) = round(bbox(i,3)*bbox(i,4));
end
[~, j] = max(area);
bbox = bbox(j,:);

% Finding ROI
x_scale = 0.25;
x_centre = round(bbox(1) + bbox(3)/2);
x_left = round(x_centre - bbox(3)*x_scale);
x_right = round(x_centre + bbox(3)*x_scale);

y_scale = 0.07;
y_centre = round(bbox(2) + bbox(4)/2);
y_centre = round(y_centre - bbox(4)*0.35);
y_top = round(y_centre - bbox(3)*y_scale);
y_bottom = round(y_centre + bbox(3)*y_scale);

ROI = [x_left,y_top,(x_right-x_left),(y_bottom-y_top)];

% Draw the returned bounding box around the detected face.
videoFrame = insertShape(videoFrame, 'Rectangle', ROI);
figure; imshow(videoFrame); title('ROI'); hold on;

% Convert the first box into a list of 4 points
% This is needed to be able to visualize the rotation of the object.
bboxPoints = bbox2points(ROI(1,:));

% Detect feature points in the face region.
points = detectMinEigenFeatures(rgb2gray(videoFrame), 'ROI', ROI,'MinQuality',0.004);

% Display the detected points.
plot(points);

% Create a point tracker and enable the bidirectional error constraint to
% make it more robust in the presence of noise and clutter.
pointTracker = vision.PointTracker('MaxBidirectionalError', 2);

% Initialize the tracker with the initial point locations and the initial
% video frame.
points = points.Location;
initialize(pointTracker, points, videoFrame);

% Make a copy of the points to be used for computing the geometric
% transformation between the points in the previous and the current frames
oldPoints = points;
NoF = VideoReader(video).NumberOfFrames;

green_average = zeros(1, NoF);

for i = 1:NoF
    % get the next frame
    videoFrame = step(videoFileReader);

    % Track the points. Note that some points may be lost.
    [points, isFound] = step(pointTracker, videoFrame);
    visiblePoints = points(isFound, :);
    oldInliers = oldPoints(isFound, :);
    
    if size(visiblePoints, 1) >= 2 % need at least 2 points
        
        % Estimate the geometric transformation between the old points
        % and the new points and eliminate outliers
        [xform, oldInliers, visiblePoints] = estimateGeometricTransform(...
            oldInliers, visiblePoints, 'similarity', 'MaxDistance', 4);
        
        % Apply the transformation to the bounding box points
        bboxPoints = transformPointsForward(xform, bboxPoints);
                
        % Insert a bounding box around the object being tracked
        bboxPolygon = reshape(bboxPoints', 1, []);     
        
        % Reset the points
        oldPoints = visiblePoints;
        setPoints(pointTracker, oldPoints);     
    end
    
    rows = round(bboxPolygon(1,2):bboxPolygon(1,6));
    columns = round(bboxPolygon(1,1):bboxPolygon(1,3));
    
    frame = readFrame(file);
    green = squeeze(frame(rows,columns,2));
%     green = medfilt2(green,[3,3]); % doesnt really affect results
    green_average(i) = mean(mean(green));
    
end

% Clean up
release(videoFileReader);
release(pointTracker);

% Plot noisy green_average channel
figure; 
plot(1:NoF,green_average);
grid on;
title('Noisy Green Average Channel');

% Filtering
fps = 30;
bpm_range = [40, 120]/60;

[b, a] = butter(2, [2*bpm_range(1)/fps, 2*bpm_range(2)/fps]);
h = fvtool(b,a);
green_filtered = filter(b, a, green_average);
signal = green_filtered(fps+1 : size(green_filtered,2));

% Plotting Filtered Channel
figure; 
plot(1:NoF,green_filtered);
grid on;
title('Filtered Green Average Channel');
