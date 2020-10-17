% X -> data to be clustered
% k -> k clusters
function [labels, KCenters] = Kmeans(X, k)
[dataSize, featureSize] = size(X);
if k > dataSize
    k = dataSize;
end

% initiate k center points
% centerArr keeps the row record of k centers
centerIdxArr = zeros(k, 1);
idx = 1;
while(idx <= k)
    if idx == 1
        centerIdxArr(idx) = randi(dataSize);
    else
        % compute the max distance of the min distances to (idx-1) points
        maxminDis = -1;
        maxminIdx = 1;
        for row = 1 : dataSize
            if isempty(find(centerIdxArr == row, 1)) == 1
                minDis = -1;
                for i = 1 : idx - 1
                    diff = X(row, :) - X(centerIdxArr(i), :);
                    dis = diff * diff';
                    if minDis == -1
                        minDis = dis;
                    else
                        if dis < minDis
                            minDis = dis;
                        end
                    end
                end
                if maxminDis == -1
                    maxminDis = minDis;
                    maxminIdx = row;
                else
                    if minDis > maxminDis
                        maxminDis = minDis;
                        maxminIdx = row;
                    end
                end
            end
        end
        centerIdxArr(idx) = maxminIdx;
    end
    idx = idx + 1;
end

% curCenterArr stores k point positions
KCenters = zeros(k, featureSize);
for idx = 1 : k
    KCenters(idx, :) = X(centerIdxArr(idx), :);
end

% compute KCenters and repeat this process until KCenters don't change
centers = zeros(dataSize, featureSize);
while 1 == 1
    newCenters = zeros(dataSize, featureSize);
    for row = 1 : dataSize
        minDis = -1;
        curCenterIdx = -1;
        for idx = 1 : k
            diff = X(row, :) - KCenters(idx, :);
            dis = diff * diff';
            if minDis == -1
                minDis = dis;
                curCenterIdx = idx;
            else
                if dis < minDis
                    minDis = dis;
                    curCenterIdx = idx;
                end
            end
        end
        newCenters(row, :) = KCenters(curCenterIdx, :);
    end
    hasChanged = ~all(centers(:) == newCenters(:));
    if hasChanged == 0
        break;
    end
    centers = newCenters;
    
    % re-compute the k centers
    for idx = 1 : k
        curCenter = KCenters(idx, :);
        sum = zeros(1, featureSize);
        counter = 0;
        for row = 1 : dataSize
            if centers(row, :) == curCenter
                sum = sum + X(row, :);
                counter = counter + 1;
            end
        end
        KCenters(idx, :) = sum ./ counter;
    end
end

% assign labels according to KCenters
labels = zeros(dataSize, 1);
for idx = 1 : k
    curCenter = KCenters(idx, :);
    for row = 1 : dataSize
        if centers(row, :) == curCenter
            labels(row) = idx;
        end
    end
end