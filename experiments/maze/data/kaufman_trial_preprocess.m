

%% Load data


folder='/Users/andrewulmer/nuin/research/mSCA/experiments/maze/data';
% 
fname='RC,2009-09-18,1-2,good-ss';
% fname='RC,2010-08-12,1-2-3,decent-ss';

load([folder '/' fname '.mat']);


%% User inputs - Matt data

dt = 10; % in ms



%% Initialize

num_trials=length(R);


%% Get information about neural data

num_nrns=length(R(1).unitRatings);

good_units = min(vertcat(R.unitRatings)) >= 1.5;

good_idx=find(good_units);

num_units=sum(good_units);

array_idx = SU.arrayLookup(good_units);

%% Extract all data
total_idx=1;

%Loop through trials and reaches
for tr=1:num_trials %Loop through trials
    if R(tr).success==0 %If it's a bad trial, skip
        continue
    end
    if R(tr).unhittable==1 %If it's a bad trial, skip
        continue
    end    
    if R(tr).unhittable==1 %If it's a bad trial, skip
        continue
    end   
    if R(tr).isConsistent==0 %If it's a bad trial, skip
        continue
    end
    if R(tr).possibleRTproblem==1 %If it's a bad trial, skip
        continue
    end   
    if R(tr).novelMaze==1 %If it's a bad trial, skip
        continue
    end
    if R(tr).primaryCondNum==0
        continue
    end
    
    
        %Find time when the target comes on for that reach
        tgt_on_time = R(tr).actualFlyAppears;       
        go_time = R(tr).actualLandingTime;
        end_time= R(tr).moveEndsTime;
        move_time= R(tr).offlineMoveOnsetTime;
        
        wdw_start=tgt_on_time-300; % -10
        wdw_end=tgt_on_time+ceil((end_time-tgt_on_time)/dt)*dt;

       %% Get target information
        tgt_id=tr; %could also use trialID
        
        %% Extract Neural Data for the reach
        
        %Break into bins of time dt
        edges=wdw_start:dt:wdw_end;     
        
        %Initialize neural data
        neural_data_temp = nan(num_units,length(edges)-1);
        
        %Change neural data from time stamps to #spikes/bin        
        for nrn_idx = 1:num_units
            nrn_num = good_idx(nrn_idx);
            neural_data_temp(nrn_idx,:) = histcounts(R(tr).unit(nrn_num).spikeTimes,edges);
        end

        
        %% Velocity
        
        vels=R(tr).handVel(edges); 
         
        %% Collect this trial's data
        
        neural_data{total_idx} = neural_data_temp;
%         kinematics{total_idx} = [x_pos, y_pos, ...
%             x_vel, y_vel, ...
%             x_acc, y_acc, ts];
%         target_id{total_idx}=tgt_id;
        target_time{total_idx}=tgt_on_time;
        tgt_time_rel{total_idx}=300;
        go_time_rel{total_idx}=go_time-tgt_on_time;
        move_time_rel{total_idx}=move_time-tgt_on_time;
        vel{total_idx}=vels;
        condition{total_idx}=R(tr).primaryCondNum;
        
        %% Put data in a struct
                 
%                
%         Data.neural_data_PMd{total_idx} = neural_data_temp_PMd;
%         Data.neural_data_M1{total_idx} = neural_data_temp_M1;
%         Data.kinematics{total_idx} = [x_pos, y_pos, ...
%             x_vel, y_vel, ...
%             x_acc, y_acc, ts];
%         
%         Data.target_id{total_idx}=tgt_id;
%         
%         % Add meta-data
%         Data.trial_num{total_idx,1} = tr;
%         Data.reach_num{total_idx,1} = 2;
%         Data.cue_on{total_idx,1} = cue_on_time;
%         Data.reach_st{total_idx,1} = reach_st;
%         Data.reach_end{total_idx,1} = reach_end;
%         Data.prev_reach_end{total_idx,1} = reach_end;
%         Data.reach_pos_st{total_idx,1} = pos_st;
%         Data.reach_pos_end{total_idx,1} = pos_end;
%         delta_pos = Data.reach_pos_end{total_idx,1} - Data.reach_pos_st{total_idx,1};
%         [Data.reach_dir{total_idx,1}, Data.reach_len{total_idx,1}] = cart2pol(delta_pos(1),delta_pos(2));
%         Data.peak_vel{total_idx,1} = NaN;
%         Data.aligned{total_idx,1} = idx_aligned;
%         

        
        
        %% Increment counter
        
        total_idx=total_idx+1;

        
        
end



%% Save

save_folder='/Users/jig289/Dropbox/MATLAB/';
% save_folder='/Users/jig289/Dropbox/MATLAB/Projects/In_Progress/Options_Coding/Miller/Data/Processed';


% save([save_folder '/' monkey '_' fname '_py.mat' ],'neural_data_PMd','neural_data_M1','kinematics','target_id','target_time','go_time_rel','move_time_rel','move_time_before_end');
save([save_folder '/kaufman_trial_py2_good_fewertr.mat' ],'neural_data','target_time','go_time_rel','move_time_rel','array_idx','vel','tgt_time_rel','condition');

