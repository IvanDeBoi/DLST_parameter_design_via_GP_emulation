clear;
clc;
figure_counter = 1;

addpath('Data');

sigma = 0.1;
use_constant_sigma = false;
use_constant_sigma = true;
optimizer = 'quasinewton';
%optimizer = 'lbfgs';

nr_of_starting_samples = 25;
nr_of_iterations = 500;
temp_diff_threshold = 5;



view_axis =[-40,40, 30];

%%
% Read data
filename = "Data\Simulator Data.csv";


data = load (filename);

diams = unique(data(:,6));

diam_emulations = [];

for index_diam = 1 : size(diams,1)-1
    diam_emulations(index_diam).diam = diams(index_diam);
    
    data_to_keep = all(data(:, 6) == diams(index_diam), 2);
    data_to_keep = data(data_to_keep, :);
    X = data_to_keep(:,[3,4,5]);
    y = data_to_keep(:,1);
    
    mins_P = [];
    rmses = [];
    eval_nrs_used = [];
    avg_sd = [];
    
    nr_best_min_so_far = -1;
    min_so_far = 10^10;
    for i = 1 : size(y, 1)
        if X(i,2) < min_so_far && y(i) > temp_diff_threshold
            nr_best_min_so_far = i;
            min_so_far = X(i,2);
        end
    end
    optimum_measured_value = y(nr_best_min_so_far, :);
    optimum_measured_position = X(nr_best_min_so_far, :);
    
    fprintf("\nPick rnd ones to start with");
    X_train = [];
    X_val = X;
    y_train = [];
    y_val = y;
    for  i = 1 : nr_of_starting_samples
        rnd_nr = randi(size(X_val, 1));
        X_train = [X_train; X_val(rnd_nr,:)];
        X_val(rnd_nr,:) = [];
        y_train = [y_train; y_val(rnd_nr,:)];
        y_val(rnd_nr,:) = [];
        eval_nrs_used = [eval_nrs_used; rnd_nr];
    end
    
    for iteration = 1 : nr_of_iterations
        fprintf("\n\nFit GP on validation data iteration %d", iteration);
        gprMdl = fitrgp(X_train,y_train ,'KernelFunction','ardsquaredexponential',...
            'verbose',0, ...
            'Optimizer',optimizer, 'Standardize', false, ...
            'SigmaLowerBound',1e-12, 'Sigma', sigma, 'ConstantSigma', use_constant_sigma);
        
        fprintf("\nMake predictions");
        [y_pred,ysd,yint] =  predict(gprMdl,X_val);
        
        fprintf("\nCalculate total error");
        rmse = 0;
        for index_val = 1 : size(X_val, 1)
            rmse = rmse + (y_pred(index_val) - y_val(index_val))^2;
        end
        rmse = rmse / size(data, 1);
        rmse = sqrt(rmse);
        rmses = [rmses;rmse];
        avg_sd = [avg_sd; mean(ysd)];
        
        if iteration < nr_of_iterations %when not last iteration
            nr = -1; %search for one with highest uncertainty
            [arg_max_value, argmax] = max(ysd);
            nr = argmax;
            X_train = [X_train; X_val(nr,:)];
            X_val(nr,:) = [];
            y_train = [y_train; y_val(nr,:)];
            y_val(nr,:) = [];
            eval_nrs_used = [eval_nrs_used; nr];
        end
    end
    
    fprintf("\n");
    
    
        fprintf("\nPlot RMSE");
        figure(figure_counter);
        clf(figure_counter);
        figure_counter = figure_counter+1;
        hold on;
            axis equal;
        grid on;
        title_string =  "RMSE";
        title(title_string);
        xlabel('nr of iterations');
        ylabel('RMSE');
        plot(rmses);
        title("Plot RMSE");
    
        fprintf("\nPlot Avg StanDev");
        figure(figure_counter);
        clf(figure_counter);
        figure_counter = figure_counter+1;
        hold on;
            axis equal;
        grid on;
        title_string =  "Avg std";
        title(title_string);
        xlabel('nr of iterations');
        ylabel('Avg std');
        plot(avg_sd);
    
        fprintf("\nPlot prediction");
        figure(figure_counter);
        clf(figure_counter);
        figure_counter = figure_counter+1;
        hold on;
        %     axis equal;
        grid on;
        c = jet;
        colormap(c);
        %Prediction
        sz = 1 + y_pred;
        scatter3(X_val(:,1), X_val(:,2),X_val(:,3), [], sz, 'filled');
        %Data
        sz = 1 + y_train;
        scatter3(X_train(:,1), X_train(:,2),X_train(:,3), [], sz,'filled'); %just for clarity
        colorbar;
        xlabel('Distance Cam Heat');
        ylabel('Heat Load');
        zlabel('Startdepth Hole');
        title("Predictions on datapoints");
        view(view_axis);
    
    
    %Use model to find optimum
    n = 20;
    range_Distance_Cam_Heat = linspace(min(X(:,1)),max(X(:,1)),n)';
    range_Heat_Load = linspace(min(X(:,2)),max(X(:,2)),n)';
    range_Startdepth_Hole = linspace(min(X(:,3)),max(X(:,3)),floor(n))';
    all_inputs = [];
    for index_dch = 1 : size(range_Distance_Cam_Heat, 1)
        for j = 1 : size(range_Heat_Load, 1)
            for k = 1 : size(range_Startdepth_Hole, 1)
                all_inputs = [all_inputs; range_Distance_Cam_Heat(index_dch), range_Heat_Load(j),range_Startdepth_Hole(k)];
            end
        end
    end
    
    [y_pred,ysd,yint] =  predict(gprMdl,all_inputs);
    all_predicted_values = y_pred;
    all_predicted_uncertainties = ysd;
    
    diam_emulations(index_diam).samples = all_inputs;
    diam_emulations(index_diam).predictions = all_predicted_values;
    
    fprintf("\n");
    fprintf("\nPlot Optima");
    figure(figure_counter);
    clf(figure_counter);
    figure_counter = figure_counter+1;
    hold on;
    grid on;  
    %Predictions
    all_inputs_bigger_than_threshold = [];
    all_inputs_smaller_than_threshold = [];
    all_outputs_bigger_than_threshold = [];
    all_outputs_smaller_than_threshold = [];
    for index_pred_val = 1 : size(all_predicted_values, 1)
        if all_predicted_values(index_pred_val) > temp_diff_threshold
            all_inputs_bigger_than_threshold = [all_inputs_bigger_than_threshold; all_inputs(index_pred_val,:)];
            all_outputs_bigger_than_threshold = [all_outputs_bigger_than_threshold; all_predicted_values(index_pred_val,:)];
        else
            all_inputs_smaller_than_threshold = [all_inputs_smaller_than_threshold; all_inputs(index_pred_val,:)];
            all_outputs_smaller_than_threshold = [all_outputs_smaller_than_threshold; all_predicted_values(index_pred_val,:)];
        end
    end
    
    c = jet;
    colormap(c);
    sz = 1 + all_outputs_bigger_than_threshold;
    scatter3(all_inputs_bigger_than_threshold(:,1), all_inputs_bigger_than_threshold(:,2),all_inputs_bigger_than_threshold(:,3), [], sz, 'filled', 'MarkerFaceAlpha',.5);
    sz = 1 + all_outputs_smaller_than_threshold;
    scatter3(all_inputs_smaller_than_threshold(:,1), all_inputs_smaller_than_threshold(:,2),all_inputs_smaller_than_threshold(:,3), [], sz,'filled', 'MarkerFaceAlpha',.5); %just for clarity
    caxis([0 50])
    xlabel('Dist[mm]');
    ylabel('P [W]');
    zlabel('Depth [mm]');
    colorbar;
    view(view_axis);
    
    for prediction_index = 1 : 100
        nr_best_min_so_far = -1;
        min_so_far = 10000;
        for i = 1 : size(all_predicted_values, 1)
            if all_inputs(i,2) < min_so_far && all_predicted_values(i) > prediction_index %% && all_predicted_uncertainties(i) < median(all_predicted_uncertainties)
                nr_best_min_so_far = i;
                min_so_far = all_inputs(i,2);
            end
        end
        if min_so_far < 10000
            mins_P = [mins_P, min_so_far];
        end
    end
    
    fprintf("\n");
    fprintf("\nPlot Interesting Range");
    figure(figure_counter);
    clf(figure_counter);
    figure_counter = figure_counter+1;
    hold on;
    grid on;
    
    lower_boundary = 5;
    upper_boundary = 25;
    
    for q = 1:size(all_outputs_bigger_than_threshold)
        if (lower_boundary <= all_outputs_bigger_than_threshold(q,:)) && (all_outputs_bigger_than_threshold(q,:) <= upper_boundary)
            idq(q,:) = 2;
        elseif  (all_outputs_bigger_than_threshold(q,:) > upper_boundary)
            idq(q,:) = 3;
        end
    end
    
    colors = [1 0 0; 0 1 0; 1 1 0];
    cmapq = colors(idq, :);
    cmapq = cmapq(1:size(all_outputs_bigger_than_threshold),:);

    scatter3(all_inputs_bigger_than_threshold(:,1), all_inputs_bigger_than_threshold(:,2),all_inputs_bigger_than_threshold(:,3), [], cmapq, 'filled', 'MarkerFaceAlpha',.5);
    scatter3(all_inputs_smaller_than_threshold(:,1), all_inputs_smaller_than_threshold(:,2),all_inputs_smaller_than_threshold(:,3), [], [1 0 0],'filled', 'MarkerFaceAlpha',.5); %just for clarity
    xlabel('Dist [mm]');
    ylabel('P [W]');
    zlabel('Depth [mm]');
    view(view_axis);
    
    cmapq = [];
    
end


%
for index_samples = 1 : 16
    nr = floor(1 + (8000-1) * rand(1,1));
    x_vals = [];
    y_vals = [];
    for index_diam = 1 : size(diam_emulations, 2)
        x_vals = [x_vals; diam_emulations(index_diam).diam];
        y_vals = [y_vals; diam_emulations(index_diam).predictions(nr)];
    end
    fprintf("\nPlot temp diff in a sample point");
    figure(figure_counter);
    clf(figure_counter);
    figure_counter = figure_counter+1;
    hold on;
    grid on;
    plot(x_vals,y_vals);
    xlabel('diam');
    ylabel('temp diff');
    title("Temp diff for rnd point");
    axis([10 26 0 80])
end



%%
for i = 1 : figure_counter
    set(figure(i),'WindowStyle','docked');
end

%%
fprintf("\n");
