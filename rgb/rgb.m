%% rgb
clc;clear all;close all;
% Create a cascade detector object.
faceDetector = vision.CascadeObjectDetector();

% Read a video frame and run the face detector.
videoFileReader = vision.VideoFileReader('rgb.mpg');
file            = VideoReader('rgb.mpg');
file2           = VideoReader('rgb.mpg');
videoFrame      = step(videoFileReader);
bbox            = step(faceDetector, videoFrame);
bbox            = round(bbox.*[1.3,1.15,0.4,0.15]);

% Draw the returned bounding box around the detected face.
videoFrame = insertShape(videoFrame, 'Rectangle', bbox);
figure; imshow(videoFrame); title('Detected face');

% Convert the first box into a list of 4 points
% This is needed to be able to visualize the rotation of the object.
bboxPoints = bbox2points(bbox(1, :));

% Detect feature points in the face region.
points = detectMinEigenFeatures(rgb2gray(videoFrame), 'ROI', bbox);

% Display the detected points.
figure, imshow(videoFrame), hold on, title('Detected features');
plot(points);

% Create a point tracker and enable the bidirectional error constraint to
% make it more robust in the presence of noise and clutter.
pointTracker = vision.PointTracker('MaxBidirectionalError', 2);

% Initialize the tracker with the initial point locations and the initial
% video frame.
points = points.Location;
initialize(pointTracker, points, videoFrame);

videoPlayer  = vision.VideoPlayer('Position',...
    [100 100 [size(videoFrame, 2), size(videoFrame, 1)]+30]);

% Make a copy of the points to be used for computing the geometric
% transformation between the points in the previous and the current frames
oldPoints = points;
NoF = file.NumberOfFrames;

green_average = zeros(1,NoF);

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
%         videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, ...
%             'LineWidth', 2);
                
        % Display tracked points
%         videoFrame = insertMarker(videoFrame, visiblePoints, '+', ...
%             'Color', 'white');       
        
        % Reset the points
        oldPoints = visiblePoints;
        setPoints(pointTracker, oldPoints);     
    end
    
    rows = round(bboxPolygon(1,2):bboxPolygon(1,6));
    columns = round(bboxPolygon(1,1):bboxPolygon(1,3));
    
    frame = readFrame(file2);
    green_average(i) = mean(mean(squeeze(frame(rows,columns,2))));
    
    % Display the annotated video frame using the video player object
    step(videoPlayer, videoFrame);
end

% Clean up
release(videoFileReader);
release(videoPlayer);
release(pointTracker);
