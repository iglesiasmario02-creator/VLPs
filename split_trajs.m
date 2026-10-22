uclear all
close all
viz = 0;
savepath = '\Projecte Laura';

% Define the pixel limits for left and right halves
left_limit = 600;

d=uigetdir(savepath);
dd = dir([d,'/*']);
dd = dd(~ismember({dd(:).name},{'.','..','scripts'}));
k=0;
for i=1:length(dd)
    
    load(fullfile(d,dd(i).name,'project analysis','TR_.mat'),'TR')
    % load('day2/project analysis/TR_.mat','TR')
    TR_left = [];
    TR_right = [];
    % Extract unique movie indices
    movie_indices = unique(TR(:,10));

    % Loop through each movie index
    for j = 1:length(movie_indices)
        k=k+1;
        
        % Get the data for the current movie index
        current_movie_index = movie_indices(j);
        TR_current = TR(TR(:,10) == current_movie_index, :);

        % Find trajectories in the left half
        left_half_indices = TR_current(:,2) <= left_limit;

        % Find trajectories in the right half
        right_half_indices = TR_current(:,2) > left_limit;

        % Extract left and right half trajectories
        TR_left_cur = TR_current(left_half_indices, :);
        TR_right_cur = TR_current(right_half_indices, :);

        % Remove trajectories that cross the boundary
        left_trajectories = unique(TR_left_cur(:,8));
        right_trajectories = unique(TR_right_cur(:,8));
        crossing_trajectories = intersect(left_trajectories, right_trajectories);

        TR_left_cur = TR_left_cur(~ismember(TR_left_cur(:,8), crossing_trajectories), :);
        TR_right_cur = TR_right_cur(~ismember(TR_right_cur(:,8), crossing_trajectories), :);

        % Correct the coordinates of the right half by subtracting 600 from X-coordinates
        TR_right_cur(:, 2) = TR_right_cur(:, 2) - left_limit;

        % % Plot the trajectories for the current movie index
        % plot(ceil(sqrt(length(movie_indices))), ceil(sqrt(length(movie_indices))), j);
        % hold on;
        if viz
            figure(k)
            % Plot left half trajectories
            left_trajectories = unique(TR_left_cur(:,8));
            for j = 1:length(left_trajectories)
                trajectory = TR_left(TR_left_cur(:,8) == left_trajectories(j), :);
                plot(trajectory(:,2), trajectory(:,1), 'Color', 'green')%, 'DisplayName', ['Left Trajectory ' num2str(left_trajectories(j))]);
                hold on
            end

            % Plot right half trajectories
            right_trajectories = unique(TR_right_cur(:,8));
            for j = 1:length(right_trajectories)
                trajectory = TR_right(TR_right_cur(:,8) == right_trajectories(j), :);
                plot(trajectory(:,2), trajectory(:,1), 'Color', 'red')%, 'DisplayName', ['Right Trajectory ' num2str(right_trajectories(j))]);
                hold on
            end

            % Customize plot
            xlabel('X Coordinate');
            ylabel('Y Coordinate');
            set(gca,"YDir","reverse")
            % title(['Movie Index: ' num2str(current_movie_index)]);
            % legend('show');
            axis equal
        end
        TR_left = [TR_left; TR_left_cur];
        TR_right = [TR_right; TR_right_cur];
        clearvars -except TR TR_left TR_right left_limit movie_indices j i viz d dd k
    end
    save(fullfile(d,dd(i).name,'project analysis','TR_left.mat'),'TR_left')
    save(fullfile(d,dd(i).name,'project analysis','TR_right.mat'),'TR_right')
    clearvars -except d dd viz left_limit i k 
end
