%% colour extraction
% abuzar nur 26/3
clc;clear all;close all;

% Create a cascade detector object.
faceDetector = vision.CascadeObjectDetector();

% Read a video frame and run the face detector.
videoFileReader = vision.VideoFileReader('face_track.mpg');
videoFrame      = step(videoFileReader);
bbox            = step(faceDetector, videoFrame);
ROI             = round(bbox.*[1.2,1.05,0.6,0.2]);

% Draw the returned bounding box around the detected face.
videoFrame = insertShape(videoFrame, 'Rectangle', bbox);
videoFrame = insertShape(videoFrame, 'Rectangle', ROI);
figure; imshow(videoFrame); title('Detected face');

% Convert the first box into a list of 4 points
% This is needed to be able to visualize the rotation of the object.
bboxPoints = bbox2points(bbox(1,:));
ROIPoints = bbox2points(ROI(1,:));

% Detect feature points in the face region.
points = detectMinEigenFeatures(rgb2gray(videoFrame), 'ROI', bbox);
pointsROI = detectMinEigenFeatures(rgb2gray(videoFrame), 'ROI', ROI);

% Display the detected points.
figure, imshow(videoFrame), hold on, title('Detected features');
plot(points);
plot(pointsROI);

% Create a point tracker and enable the bidirectional error constraint to
% make it more robust in the presence of noise and clutter.
pointTracker = vision.PointTracker('MaxBidirectionalError', 2);
pointTrackerROI = vision.PointTracker('MaxBidirectionalError', 2);

% Initialize the tracker with the initial point locations and the initial
% video frame.
points = points.Location;
pointsROI = pointsROI.Location;
initialize(pointTracker, points, videoFrame);
initialize(pointTrackerROI, pointsROI, videoFrame);

videoPlayer  = vision.VideoPlayer('Position',...
    [100 100 [size(videoFrame, 2), size(videoFrame, 1)]+30]);

% Make a copy of the points to be used for computing the geometric
% transformation between the points in the previous and the current frames
oldPoints = points;
oldPointsROI = pointsROI;

while ~isDone(videoFileReader)
    % get the next frame
    videoFrame = step(videoFileReader);

    % Track the points. Note that some points may be lost.
    [points, isFound] = step(pointTracker, videoFrame);
    [pointsROI, isFoundROI] = step(pointTrackerROI, videoFrame);
    visiblePoints = points(isFound, :); 
    visiblePointsROI = pointsROI(isFoundROI, :);
    oldInliers = oldPoints(isFound, :);
    oldInliersROI = oldPointsROI(isFoundROI, :);
    
    if size(visiblePoints, 1) >= 2 % need at least 2 points
        
        % Estimate the geometric transformation between the old points
        % and the new points and eliminate outliers
        [xform, oldInliers, visiblePoints] = estimateGeometricTransform(...
            oldInliers, visiblePoints, 'similarity', 'MaxDistance', 4);
        
        [xformROI, oldInliersROI, visiblePointsROI] = estimateGeometricTransform(...
            oldInliersROI, visiblePointsROI, 'similarity', 'MaxDistance', 4);
        
        % Apply the transformation to the bounding box points
        bboxPoints = transformPointsForward(xform, bboxPoints);
        ROIPoints = transformPointsForward(xformROI, ROIPoints);
                
        % Insert a bounding box around the object being tracked
        bboxPolygon = reshape(bboxPoints', 1, []);
        ROIPolygon = reshape(ROIPoints', 1, []);
        videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, ...
            'LineWidth', 1);
        videoFrame = insertShape(videoFrame, 'Polygon', ROIPolygon, ...
            'LineWidth', 3,'Color', 'white');
                
        % Display tracked points
        videoFrame = insertMarker(videoFrame, visiblePoints, '+', ...
            'Color', 'white');
        videoFrame = insertMarker(videoFrame, visiblePointsROI, '+', ...
            'Color', 'green'); 
        
        % Reset the points
        oldPoints = visiblePoints;
        oldPointsROI = visiblePointsROI;
        setPoints(pointTracker, oldPoints);
        setPoints(pointTrackerROI, oldPointsROI);
        
    end
    
    % Display the annotated video frame using the video player object
    step(videoPlayer, videoFrame);
end

% Clean up
release(videoFileReader);
release(videoPlayer);
release(pointTracker);
release(pointTrackerROI);

