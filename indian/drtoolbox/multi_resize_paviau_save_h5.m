close all;
clear all;
clc;
addpath('drtoolbox');
addpath('drtoolbox/gui');
addpath('drtoolbox/techniques');

savepath_train3 = 'HSI3/train.h5';
savepath_test3  = 'HSI3/';

count_batch=1;
pad_size1=1;
num_bands=3;
input_size1=2*pad_size1+1;
no_classes=16;
%% number of training samples per class

per_class_num=ceil([46,1428,830,237,483,730,28,478,20,972,2455,593,205,1265,386,93;]*0.5)       % 1% sampling. test: 42330=30*1411; train: 446

load Indian_pines_corrected.mat;
img=indian_pines_corrected;
[I_row,I_line,I_high] = size(img);
img=reshape(img,I_row*I_line,I_high);

%%%%%%% PCA %%%%%%
im=img;
im=compute_mapping(im,'PCA',num_bands);
im = mat2gray(im);
im =reshape(im,[I_row,I_line,num_bands]);

%%%%%  scale the image from -1 to 1
im=reshape(im,I_row*I_line,num_bands);
[im ] = scale_func(im);
im =reshape(im,[I_row,I_line,num_bands]);
%%%%%% extending the image %%%%%%%%
im_extend=padarray(im,[pad_size1,pad_size1],'symmetric');

load Indian_pines_gt.mat;
Train_Label = [];
Train_index = [];
train_data=[];
test_data=[];
train_label=[];
test_label=[];
train_index=[];
test_index=[];
index_len=[];

% index_len=[];
for ii = 1: no_classes
    
   index_ii =  find(indian_pines_gt == ii)';
   index_len=[index_len length(index_ii)];
   labeled_len=ceil(index_len*0.07);
   rand_order=randperm(length(index_ii));
   class_ii = ones(1,length(index_ii))* ii;
   Train_Label = [Train_Label class_ii];
   Train_index = [Train_index index_ii]; 
   
   num_train=per_class_num(ii);
 % num_train=floor(length(index_ii)*percent);
   train_ii=rand_order(:,1:num_train);
   train_index=[train_index index_ii(train_ii)];
%   train_index=[train_index index_ii(train_ii)];
   
   test_index_temp=index_ii;
   test_index_temp(:,train_ii)=[];
   test_index=[test_index test_index_temp];
%%   test_index=[test_index test_index_temp];
   
   train_label=[train_label class_ii(:,1:num_train)];
   test_label=[test_label class_ii(num_train+1:end)];
   
end

%%%%%  generate training samples and labels%%%%%%%%%%%%%%%%%%%%%%
TRAIN_DATA=zeros(31,31,num_bands,length(train_label));
TRAIN_LABEL=train_label-1;
count=0;
    
        for j=1:length(train_index)
            count=count+1;

            img_data=[];
            X=mod(train_index(j),I_row);
            Y=ceil(train_index(j)/I_row);
            if X==0
               X=I_row;
            end
            if Y==0
              Y=I_line;
            end
            X_new = X+pad_size1;
            Y_new = Y+pad_size1;
            X_range = [X_new-pad_size1 : X_new+pad_size1];
      
            Y_range = [Y_new-pad_size1 : Y_new+pad_size1]; 

            img_data=im_extend(X_range,Y_range,:);
            
            img_data=imresize(img_data,[31 31],'bicubic');
            
            TRAIN_DATA(:,:,:,count)=img_data;
            
        
        end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%    write train.hdf5    %%%%%%%%%%%%%
order1 = randperm(count);
TRAIN_INDEX=train_index(:,order1);
TRAIN_DATA=TRAIN_DATA(:,:,:,order1);
TRAIN_DATA=permute(TRAIN_DATA,[2 1 3 4]);
TRAIN_LABEL=TRAIN_LABEL(:,order1);
data=TRAIN_DATA;
label=TRAIN_LABEL;
chunksz = 100;
created_flag = false;
totalct = 0;

for batchno = 1:ceil(count/chunksz)
    last_read=(batchno-1)*chunksz;
    if batchno*chunksz>count
        batchdata = data(:,:,:,last_read+1:end); 
        batchlabs = label(:,last_read+1:end); 
    else
        batchdata = data(:,:,:,last_read+1:last_read+chunksz); 
        batchlabs = label(:,last_read+1:last_read+chunksz);
    end

    startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,totalct+1]);
    curr_dat_sz = store2hdf5(savepath_train3, batchdata, batchlabs, ~created_flag, startloc, chunksz); 
    created_flag = true;
    totalct = curr_dat_sz(end);
end
h5disp(savepath_train3);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%% generate testing samples and labels  %%%%%%%%%%%%%%%%%%
TEST_DATA=zeros(31,31,num_bands,length(test_label));
TEST_LABEL=test_label-1;
count=0;
   
        for j=1:length(test_index)
            count=count+1;

            img_data=[];
            X=mod(test_index(j),I_row);
            Y=ceil(test_index(j)/I_row);
            if X==0
               X=I_row;
            end
            if Y==0
              Y=I_line;
            end
            X_new = X+pad_size1;
            Y_new = Y+pad_size1;
            X_range = [X_new-pad_size1 : X_new+pad_size1];
      
            Y_range = [Y_new-pad_size1 : Y_new+pad_size1]; 

            img_data=im_extend(X_range,Y_range,:);
            
            img_data=imresize(img_data,[31 31],'bicubic');
            
            TEST_DATA(:,:,:,count)=img_data;
            
        
        end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%   write test.hdf5 %%%%%%%%%%%%%%%%%%%%%%%%%%%
order2 = randperm(count);
TEST_INDEX=test_index(:,order2);
TEST_DATA=TEST_DATA(:,:,:,order2);
TEST_DATA=permute(TEST_DATA,[2 1 3 4]);
TEST_LABEL=TEST_LABEL(:,order2);
for k=1:count_batch
    count=length(TEST_LABEL)/count_batch;
    data=TEST_DATA(:,:,:,(k-1)*count+1:k*count);
    label=TEST_LABEL(:,(k-1)*count+1:k*count);
    chunksz = 100;
    created_flag = false;
    totalct = 0;
    savepath_test_temp=strcat(savepath_test3,'test',num2str(k),'.h5');
    for batchno = 1:ceil(count/chunksz)
        last_read=(batchno-1)*chunksz;
        if batchno*chunksz>count
            batchdata = data(:,:,:,last_read+1:end); 
            batchlabs = label(:,last_read+1:end); 
        else
            batchdata = data(:,:,:,last_read+1:last_read+chunksz); 
            batchlabs = label(:,last_read+1:last_read+chunksz);
        end

        startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,totalct+1]);
        curr_dat_sz = store2hdf5(savepath_test_temp, batchdata, batchlabs, ~created_flag, startloc, chunksz); 
        created_flag = true;
        totalct = curr_dat_sz(end);
    end
    h5disp(savepath_test_temp);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

save HSI3/data.mat TRAIN_INDEX TEST_INDEX TEST_LABEL TRAIN_LABEL train_index test_index train_label test_label





savepath_train5 = 'HSI5/train.h5';
savepath_test5  = 'HSI5/';


pad_size2=2;
input_size2=2*pad_size2+1;
%% number of training samples per class

%per_class_num=[1,15,9,3,5,8,1,5,1,10,24,6,3,13,4,1];                 % 1% sampling. test: 10140=20*507' train: 109 
%per_class_num=[3,72,42,12,25,37,2,24,1,49,122,30,11,64,20,5];        % 5%  sampling. test: 9730=10*973; train: 519
%per_class_num=[5,100,59,17,34,52,3,34,3,69,172,42,15,89,28,7];      % 7% sampling. test: 9520=20*476; train: 729
per_class_num=[5,143,83,24,49,73,3,48,2,98,245,60,21,126,39,10];     % 10% sampling. test: 9220=20*461; train: 1029
%per_class_num=[7,214,124,36,73,110,5,72,3,146,368,89,31,189,58,14];   % 15% samling.  test: 8710=10*871; train: 1539
%per_class_num=[10,285,165,48,97,145,6,96,4,194,490,119,41,252,78,19];       % 20% sampling. test: 8200=50*164; train: 2049


%%%%%% extending the image %%%%%%%%
im_extend=padarray(im,[pad_size2,pad_size2],'symmetric');




%%%%%  generate training samples and labels%%%%%%%%%%%%%%%%%%%%%%
TRAIN_DATA=zeros(31,31,num_bands,length(train_label));
TRAIN_LABEL=train_label-1;
count=0;
    
        for j=1:length(train_index)
            count=count+1;

            img_data=[];
            X=mod(train_index(j),I_row);
            Y=ceil(train_index(j)/I_row);
            if X==0
               X=I_row;
            end
            if Y==0
              Y=I_line;
            end
            X_new = X+pad_size2;
            Y_new = Y+pad_size2;
            X_range = [X_new-pad_size2 : X_new+pad_size2];
      
            Y_range = [Y_new-pad_size2 : Y_new+pad_size2]; 

            img_data=im_extend(X_range,Y_range,:);
            
            img_data=imresize(img_data,[31 31],'bicubic');
            
            TRAIN_DATA(:,:,:,count)=img_data;
            
        
        end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%    write train.hdf5    %%%%%%%%%%%%%
TRAIN_INDEX=train_index(:,order1);
TRAIN_DATA=TRAIN_DATA(:,:,:,order1);
TRAIN_DATA=permute(TRAIN_DATA,[2 1 3 4]);
TRAIN_LABEL=TRAIN_LABEL(:,order1);
data=TRAIN_DATA;
label=TRAIN_LABEL;
chunksz = 100;
created_flag = false;
totalct = 0;

for batchno = 1:ceil(count/chunksz)
    last_read=(batchno-1)*chunksz;
    if batchno*chunksz>count
        batchdata = data(:,:,:,last_read+1:end); 
        batchlabs = label(:,last_read+1:end); 
    else
        batchdata = data(:,:,:,last_read+1:last_read+chunksz); 
        batchlabs = label(:,last_read+1:last_read+chunksz);
    end

    startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,totalct+1]);
    curr_dat_sz = store2hdf5(savepath_train5, batchdata, batchlabs, ~created_flag, startloc, chunksz); 
    created_flag = true;
    totalct = curr_dat_sz(end);
end
h5disp(savepath_train5);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%% generate testing samples and labels  %%%%%%%%%%%%%%%%%%
TEST_DATA=zeros(31,31,num_bands,length(test_label));
TEST_LABEL=test_label-1;
count=0;
   
        for j=1:length(test_index)
            count=count+1;

            img_data=[];
            X=mod(test_index(j),I_row);
            Y=ceil(test_index(j)/I_row);
            if X==0
               X=I_row;
            end
            if Y==0
              Y=I_line;
            end
            X_new = X+pad_size2;
            Y_new = Y+pad_size2;
            X_range = [X_new-pad_size2 : X_new+pad_size2];
      
            Y_range = [Y_new-pad_size2 : Y_new+pad_size2]; 

            img_data=im_extend(X_range,Y_range,:);
            
            img_data=imresize(img_data,[31 31],'bicubic');
            
            TEST_DATA(:,:,:,count)=img_data;
            
        
        end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%   write test.hdf5 %%%%%%%%%%%%%%%%%%%%%%%%%%%
TEST_INDEX=test_index(:,order2);
TEST_DATA=TEST_DATA(:,:,:,order2);
TEST_DATA=permute(TEST_DATA,[2 1 3 4]);
TEST_LABEL=TEST_LABEL(:,order2);
for k=1:count_batch
    count=length(TEST_LABEL)/count_batch;
    data=TEST_DATA(:,:,:,(k-1)*count+1:k*count);
    label=TEST_LABEL(:,(k-1)*count+1:k*count);
    chunksz = 100;
    created_flag = false;
    totalct = 0;
    savepath_test_temp=strcat(savepath_test5,'test',num2str(k),'.h5');
    for batchno = 1:ceil(count/chunksz)
        last_read=(batchno-1)*chunksz;
        if batchno*chunksz>count
            batchdata = data(:,:,:,last_read+1:end); 
            batchlabs = label(:,last_read+1:end); 
        else
            batchdata = data(:,:,:,last_read+1:last_read+chunksz); 
            batchlabs = label(:,last_read+1:last_read+chunksz);
        end

        startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,totalct+1]);
        curr_dat_sz = store2hdf5(savepath_test_temp, batchdata, batchlabs, ~created_flag, startloc, chunksz); 
        created_flag = true;
        totalct = curr_dat_sz(end);
    end
    h5disp(savepath_test_temp);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

save HSI5/data.mat TRAIN_INDEX TEST_INDEX TEST_LABEL TRAIN_LABEL train_index test_index train_label test_label






savepath_train7 = 'HSI7/train.h5';
savepath_test7  = 'HSI7/';


pad_size3=3;
input_size3=2*pad_size3+1;
%% number of training samples per class

%per_class_num=[1,15,9,3,5,8,1,5,1,10,24,6,3,13,4,1];                 % 1% sampling. test: 10140=20*507' train: 109 
%per_class_num=[3,72,42,12,25,37,2,24,1,49,122,30,11,64,20,5];        % 5%  sampling. test: 9730=10*973; train: 519
%per_class_num=[5,100,59,17,34,52,3,34,3,69,172,42,15,89,28,7];      % 7% sampling. test: 9520=20*476; train: 729
per_class_num=[5,143,83,24,49,73,3,48,2,98,245,60,21,126,39,10];     % 10% sampling. test: 9220=20*461; train: 1029
%per_class_num=[7,214,124,36,73,110,5,72,3,146,368,89,31,189,58,14];   % 15% samling.  test: 8710=10*871; train: 1539
%per_class_num=[10,285,165,48,97,145,6,96,4,194,490,119,41,252,78,19];       % 20% sampling. test: 8200=50*164; train: 2049


%%%%%% extending the image %%%%%%%%
im_extend=padarray(im,[pad_size3,pad_size3],'symmetric');




%%%%%  generate training samples and labels%%%%%%%%%%%%%%%%%%%%%%
TRAIN_DATA=zeros(31,31,num_bands,length(train_label));
TRAIN_LABEL=train_label-1;
count=0;
    
        for j=1:length(train_index)
            count=count+1;

            img_data=[];
            X=mod(train_index(j),I_row);
            Y=ceil(train_index(j)/I_row);
            if X==0
               X=I_row;
            end
            if Y==0
              Y=I_line;
            end
            X_new = X+pad_size3;
            Y_new = Y+pad_size3;
            X_range = [X_new-pad_size3 : X_new+pad_size3];
      
            Y_range = [Y_new-pad_size3 : Y_new+pad_size3]; 

            img_data=im_extend(X_range,Y_range,:);
            
            img_data=imresize(img_data,[31 31],'bicubic');
            
            TRAIN_DATA(:,:,:,count)=img_data;
            
        
        end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%    write train.hdf5    %%%%%%%%%%%%%
TRAIN_INDEX=train_index(:,order1);
TRAIN_DATA=TRAIN_DATA(:,:,:,order1);
TRAIN_DATA=permute(TRAIN_DATA,[2 1 3 4]);
TRAIN_LABEL=TRAIN_LABEL(:,order1);
data=TRAIN_DATA;
label=TRAIN_LABEL;
chunksz = 100;
created_flag = false;
totalct = 0;

for batchno = 1:ceil(count/chunksz)
    last_read=(batchno-1)*chunksz;
    if batchno*chunksz>count
        batchdata = data(:,:,:,last_read+1:end); 
        batchlabs = label(:,last_read+1:end); 
    else
        batchdata = data(:,:,:,last_read+1:last_read+chunksz); 
        batchlabs = label(:,last_read+1:last_read+chunksz);
    end

    startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,totalct+1]);
    curr_dat_sz = store2hdf5(savepath_train7, batchdata, batchlabs, ~created_flag, startloc, chunksz); 
    created_flag = true;
    totalct = curr_dat_sz(end);
end
h5disp(savepath_train7);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%% generate testing samples and labels  %%%%%%%%%%%%%%%%%%
TEST_DATA=zeros(31,31,num_bands,length(test_label));
TEST_LABEL=test_label-1;
count=0;
   
        for j=1:length(test_index)
            count=count+1;

            img_data=[];
            X=mod(test_index(j),I_row);
            Y=ceil(test_index(j)/I_row);
            if X==0
               X=I_row;
            end
            if Y==0
              Y=I_line;
            end
            X_new = X+pad_size3;
            Y_new = Y+pad_size3;
            X_range = [X_new-pad_size3 : X_new+pad_size3];
      
            Y_range = [Y_new-pad_size3 : Y_new+pad_size3]; 

            img_data=im_extend(X_range,Y_range,:);
            
            img_data=imresize(img_data,[31 31],'bicubic');
            
            TEST_DATA(:,:,:,count)=img_data;
            
        
        end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%   write test.hdf5 %%%%%%%%%%%%%%%%%%%%%%%%%%%
TEST_INDEX=test_index(:,order2);
TEST_DATA=TEST_DATA(:,:,:,order2);
TEST_DATA=permute(TEST_DATA,[2 1 3 4]);
TEST_LABEL=TEST_LABEL(:,order2);
for k=1:count_batch
    count=length(TEST_LABEL)/count_batch;
    data=TEST_DATA(:,:,:,(k-1)*count+1:k*count);
    label=TEST_LABEL(:,(k-1)*count+1:k*count);
    chunksz = 100;
    created_flag = false;
    totalct = 0;
    savepath_test_temp=strcat(savepath_test7,'test',num2str(k),'.h5');
    for batchno = 1:ceil(count/chunksz)
        last_read=(batchno-1)*chunksz;
        if batchno*chunksz>count
            batchdata = data(:,:,:,last_read+1:end); 
            batchlabs = label(:,last_read+1:end); 
        else
            batchdata = data(:,:,:,last_read+1:last_read+chunksz); 
            batchlabs = label(:,last_read+1:last_read+chunksz);
        end

        startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,totalct+1]);
        curr_dat_sz = store2hdf5(savepath_test_temp, batchdata, batchlabs, ~created_flag, startloc, chunksz); 
        created_flag = true;
        totalct = curr_dat_sz(end);
    end
    h5disp(savepath_test_temp);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

save HSI7/data.mat TRAIN_INDEX TEST_INDEX TEST_LABEL TRAIN_LABEL train_index test_index train_label test_label






savepath_train9 = 'HSI9/train.h5';
savepath_test9  = 'HSI9/';


pad_size4=4;
input_size4=2*pad_size4+1;
%% number of training samples per class

%per_class_num=[1,15,9,3,5,8,1,5,1,10,24,6,3,13,4,1];                 % 1% sampling. test: 10140=20*507' train: 109 
%per_class_num=[3,72,42,12,25,37,2,24,1,49,122,30,11,64,20,5];        % 5%  sampling. test: 9730=10*973; train: 519
%per_class_num=[5,100,59,17,34,52,3,34,3,69,172,42,15,89,28,7];      % 7% sampling. test: 9520=20*476; train: 729
per_class_num=[5,143,83,24,49,73,3,48,2,98,245,60,21,126,39,10];     % 10% sampling. test: 9220=20*461; train: 1029
%per_class_num=[7,214,124,36,73,110,5,72,3,146,368,89,31,189,58,14];   % 15% samling.  test: 8710=10*871; train: 1539
%per_class_num=[10,285,165,48,97,145,6,96,4,194,490,119,41,252,78,19];       % 20% sampling. test: 8200=50*164; train: 2049


%%%%%% extending the image %%%%%%%%
im_extend=padarray(im,[pad_size4,pad_size4],'symmetric');




%%%%%  generate training samples and labels%%%%%%%%%%%%%%%%%%%%%%
TRAIN_DATA=zeros(31,31,num_bands,length(train_label));
TRAIN_LABEL=train_label-1;
count=0;
    
        for j=1:length(train_index)
            count=count+1;

            img_data=[];
            X=mod(train_index(j),I_row);
            Y=ceil(train_index(j)/I_row);
            if X==0
               X=I_row;
            end
            if Y==0
              Y=I_line;
            end
            X_new = X+pad_size4;
            Y_new = Y+pad_size4;
            X_range = [X_new-pad_size4 : X_new+pad_size4];
      
            Y_range = [Y_new-pad_size4 : Y_new+pad_size4]; 

            img_data=im_extend(X_range,Y_range,:);
            
            img_data=imresize(img_data,[31 31],'bicubic');
            
            TRAIN_DATA(:,:,:,count)=img_data;
            
        
        end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%    write train.hdf5    %%%%%%%%%%%%%
TRAIN_INDEX=train_index(:,order1);
TRAIN_DATA=TRAIN_DATA(:,:,:,order1);
TRAIN_DATA=permute(TRAIN_DATA,[2 1 3 4]);
TRAIN_LABEL=TRAIN_LABEL(:,order1);
data=TRAIN_DATA;
label=TRAIN_LABEL;
chunksz = 100;
created_flag = false;
totalct = 0;

for batchno = 1:ceil(count/chunksz)
    last_read=(batchno-1)*chunksz;
    if batchno*chunksz>count
        batchdata = data(:,:,:,last_read+1:end); 
        batchlabs = label(:,last_read+1:end); 
    else
        batchdata = data(:,:,:,last_read+1:last_read+chunksz); 
        batchlabs = label(:,last_read+1:last_read+chunksz);
    end

    startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,totalct+1]);
    curr_dat_sz = store2hdf5(savepath_train9, batchdata, batchlabs, ~created_flag, startloc, chunksz); 
    created_flag = true;
    totalct = curr_dat_sz(end);
end
h5disp(savepath_train9);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%% generate testing samples and labels  %%%%%%%%%%%%%%%%%%
TEST_DATA=zeros(31,31,num_bands,length(test_label));
TEST_LABEL=test_label-1;
count=0;
   
        for j=1:length(test_index)
            count=count+1;

            img_data=[];
            X=mod(test_index(j),I_row);
            Y=ceil(test_index(j)/I_row);
            if X==0
               X=I_row;
            end
            if Y==0
              Y=I_line;
            end
            X_new = X+pad_size4;
            Y_new = Y+pad_size4;
            X_range = [X_new-pad_size4 : X_new+pad_size4];
      
            Y_range = [Y_new-pad_size4 : Y_new+pad_size4]; 

            img_data=im_extend(X_range,Y_range,:);
            
            img_data=imresize(img_data,[31 31],'bicubic');
            
            TEST_DATA(:,:,:,count)=img_data;
            
        
        end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%   write test.hdf5 %%%%%%%%%%%%%%%%%%%%%%%%%%%
TEST_INDEX=test_index(:,order2);
TEST_DATA=TEST_DATA(:,:,:,order2);
TEST_DATA=permute(TEST_DATA,[2 1 3 4]);
TEST_LABEL=TEST_LABEL(:,order2);
for k=1:count_batch
    count=length(TEST_LABEL)/count_batch;
    data=TEST_DATA(:,:,:,(k-1)*count+1:k*count);
    label=TEST_LABEL(:,(k-1)*count+1:k*count);
    chunksz = 100;
    created_flag = false;
    totalct = 0;
    savepath_test_temp=strcat(savepath_test9,'test',num2str(k),'.h5');
    for batchno = 1:ceil(count/chunksz)
        last_read=(batchno-1)*chunksz;
        if batchno*chunksz>count
            batchdata = data(:,:,:,last_read+1:end); 
            batchlabs = label(:,last_read+1:end); 
        else
            batchdata = data(:,:,:,last_read+1:last_read+chunksz); 
            batchlabs = label(:,last_read+1:last_read+chunksz);
        end

        startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,totalct+1]);
        curr_dat_sz = store2hdf5(savepath_test_temp, batchdata, batchlabs, ~created_flag, startloc, chunksz); 
        created_flag = true;
        totalct = curr_dat_sz(end);
    end
    h5disp(savepath_test_temp);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

save HSI9/data.mat TRAIN_INDEX TEST_INDEX TEST_LABEL TRAIN_LABEL train_index test_index train_label test_label










savepath_train11 = 'HSI11/train.h5';
savepath_test11  = 'HSI11/';


pad_size5=5;
input_size5=2*pad_size5+1;
%% number of training samples per class

%per_class_num=[1,15,9,3,5,8,1,5,1,10,24,6,3,13,4,1];                 % 1% sampling. test: 10140=20*507' train: 109 
%per_class_num=[3,72,42,12,25,37,2,24,1,49,122,30,11,64,20,5];        % 5%  sampling. test: 9730=10*973; train: 519
%per_class_num=[5,100,59,17,34,52,3,34,3,69,172,42,15,89,28,7];      % 7% sampling. test: 9520=20*476; train: 729
per_class_num=[5,143,83,24,49,73,3,48,2,98,245,60,21,126,39,10];     % 10% sampling. test: 9220=20*461; train: 1029
%per_class_num=[7,214,124,36,73,110,5,72,3,146,368,89,31,189,58,14];   % 15% samling.  test: 8710=10*871; train: 1539
%per_class_num=[10,285,165,48,97,145,6,96,4,194,490,119,41,252,78,19];       % 20% sampling. test: 8200=50*164; train: 2049


%%%%%% extending the image %%%%%%%%
im_extend=padarray(im,[pad_size5,pad_size5],'symmetric');




%%%%%  generate training samples and labels%%%%%%%%%%%%%%%%%%%%%%
TRAIN_DATA=zeros(31,31,num_bands,length(train_label));
TRAIN_LABEL=train_label-1;
count=0;
    
        for j=1:length(train_index)
            count=count+1;

            img_data=[];
            X=mod(train_index(j),I_row);
            Y=ceil(train_index(j)/I_row);
            if X==0
               X=I_row;
            end
            if Y==0
              Y=I_line;
            end
            X_new = X+pad_size5;
            Y_new = Y+pad_size5;
            X_range = [X_new-pad_size5 : X_new+pad_size5];
      
            Y_range = [Y_new-pad_size5 : Y_new+pad_size5]; 

            img_data=im_extend(X_range,Y_range,:);
            
            img_data=imresize(img_data,[31 31],'bicubic');
            
            TRAIN_DATA(:,:,:,count)=img_data;
            
        
        end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%    write train.hdf5    %%%%%%%%%%%%%
TRAIN_INDEX=train_index(:,order1);
TRAIN_DATA=TRAIN_DATA(:,:,:,order1);
TRAIN_DATA=permute(TRAIN_DATA,[2 1 3 4]);
TRAIN_LABEL=TRAIN_LABEL(:,order1);
data=TRAIN_DATA;
label=TRAIN_LABEL;
chunksz = 100;
created_flag = false;
totalct = 0;

for batchno = 1:ceil(count/chunksz)
    last_read=(batchno-1)*chunksz;
    if batchno*chunksz>count
        batchdata = data(:,:,:,last_read+1:end); 
        batchlabs = label(:,last_read+1:end); 
    else
        batchdata = data(:,:,:,last_read+1:last_read+chunksz); 
        batchlabs = label(:,last_read+1:last_read+chunksz);
    end

    startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,totalct+1]);
    curr_dat_sz = store2hdf5(savepath_train11, batchdata, batchlabs, ~created_flag, startloc, chunksz); 
    created_flag = true;
    totalct = curr_dat_sz(end);
end
h5disp(savepath_train11);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%% generate testing samples and labels  %%%%%%%%%%%%%%%%%%
TEST_DATA=zeros(31,31,num_bands,length(test_label));
TEST_LABEL=test_label-1;
count=0;
   
        for j=1:length(test_index)
            count=count+1;

            img_data=[];
            X=mod(test_index(j),I_row);
            Y=ceil(test_index(j)/I_row);
            if X==0
               X=I_row;
            end
            if Y==0
              Y=I_line;
            end
            X_new = X+pad_size5;
            Y_new = Y+pad_size5;
            X_range = [X_new-pad_size5 : X_new+pad_size5];
      
            Y_range = [Y_new-pad_size5 : Y_new+pad_size5]; 

            img_data=im_extend(X_range,Y_range,:);
            
            img_data=imresize(img_data,[31 31],'bicubic');
            
            TEST_DATA(:,:,:,count)=img_data;
            
        
        end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%   write test.hdf5 %%%%%%%%%%%%%%%%%%%%%%%%%%%
TEST_INDEX=test_index(:,order2);
TEST_DATA=TEST_DATA(:,:,:,order2);
TEST_DATA=permute(TEST_DATA,[2 1 3 4]);
TEST_LABEL=TEST_LABEL(:,order2);
for k=1:count_batch
    count=length(TEST_LABEL)/count_batch;
    data=TEST_DATA(:,:,:,(k-1)*count+1:k*count);
    label=TEST_LABEL(:,(k-1)*count+1:k*count);
    chunksz = 100;
    created_flag = false;
    totalct = 0;
    savepath_test_temp=strcat(savepath_test11,'test',num2str(k),'.h5');
    for batchno = 1:ceil(count/chunksz)
        last_read=(batchno-1)*chunksz;
        if batchno*chunksz>count
            batchdata = data(:,:,:,last_read+1:end); 
            batchlabs = label(:,last_read+1:end); 
        else
            batchdata = data(:,:,:,last_read+1:last_read+chunksz); 
            batchlabs = label(:,last_read+1:last_read+chunksz);
        end

        startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,totalct+1]);
        curr_dat_sz = store2hdf5(savepath_test_temp, batchdata, batchlabs, ~created_flag, startloc, chunksz); 
        created_flag = true;
        totalct = curr_dat_sz(end);
    end
    h5disp(savepath_test_temp);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

save HSI11/data.mat TRAIN_INDEX TEST_INDEX TEST_LABEL TRAIN_LABEL train_index test_index train_label test_label






savepath_train13 = 'HSI13/train.h5';
savepath_test13  = 'HSI13/';


pad_size6=6;
input_size6=2*pad_size6+1;
%% number of training samples per class

%per_class_num=[1,15,9,3,5,8,1,5,1,10,24,6,3,13,4,1];                 % 1% sampling. test: 10140=20*507' train: 109 
%per_class_num=[3,72,42,12,25,37,2,24,1,49,122,30,11,64,20,5];        % 5%  sampling. test: 9730=10*973; train: 519
%per_class_num=[5,100,59,17,34,52,3,34,3,69,172,42,15,89,28,7];      % 7% sampling. test: 9520=20*476; train: 729
per_class_num=[5,143,83,24,49,73,3,48,2,98,245,60,21,126,39,10];     % 10% sampling. test: 9220=20*461; train: 1029
%per_class_num=[7,214,124,36,73,110,5,72,3,146,368,89,31,189,58,14];   % 15% samling.  test: 8710=10*871; train: 1539
%per_class_num=[10,285,165,48,97,145,6,96,4,194,490,119,41,252,78,19];       % 20% sampling. test: 8200=50*164; train: 2049


%%%%%% extending the image %%%%%%%%
im_extend=padarray(im,[pad_size6,pad_size6],'symmetric');




%%%%%  generate training samples and labels%%%%%%%%%%%%%%%%%%%%%%
TRAIN_DATA=zeros(31,31,num_bands,length(train_label));
TRAIN_LABEL=train_label-1;
count=0;
    
        for j=1:length(train_index)
            count=count+1;

            img_data=[];
            X=mod(train_index(j),I_row);
            Y=ceil(train_index(j)/I_row);
            if X==0
               X=I_row;
            end
            if Y==0
              Y=I_line;
            end
            X_new = X+pad_size6;
            Y_new = Y+pad_size6;
            X_range = [X_new-pad_size6 : X_new+pad_size6];
      
            Y_range = [Y_new-pad_size6 : Y_new+pad_size6]; 

            img_data=im_extend(X_range,Y_range,:);
            
            img_data=imresize(img_data,[31 31],'bicubic');
            
            TRAIN_DATA(:,:,:,count)=img_data;
            
        
        end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%    write train.hdf5    %%%%%%%%%%%%%
TRAIN_INDEX=train_index(:,order1);
TRAIN_DATA=TRAIN_DATA(:,:,:,order1);
TRAIN_DATA=permute(TRAIN_DATA,[2 1 3 4]);
TRAIN_LABEL=TRAIN_LABEL(:,order1);
data=TRAIN_DATA;
label=TRAIN_LABEL;
chunksz = 100;
created_flag = false;
totalct = 0;

for batchno = 1:ceil(count/chunksz)
    last_read=(batchno-1)*chunksz;
    if batchno*chunksz>count
        batchdata = data(:,:,:,last_read+1:end); 
        batchlabs = label(:,last_read+1:end); 
    else
        batchdata = data(:,:,:,last_read+1:last_read+chunksz); 
        batchlabs = label(:,last_read+1:last_read+chunksz);
    end

    startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,totalct+1]);
    curr_dat_sz = store2hdf5(savepath_train13, batchdata, batchlabs, ~created_flag, startloc, chunksz); 
    created_flag = true;
    totalct = curr_dat_sz(end);
end
h5disp(savepath_train13);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%% generate testing samples and labels  %%%%%%%%%%%%%%%%%%
TEST_DATA=zeros(31,31,num_bands,length(test_label));
TEST_LABEL=test_label-1;
count=0;
   
        for j=1:length(test_index)
            count=count+1;

            img_data=[];
            X=mod(test_index(j),I_row);
            Y=ceil(test_index(j)/I_row);
            if X==0
               X=I_row;
            end
            if Y==0
              Y=I_line;
            end
            X_new = X+pad_size6;
            Y_new = Y+pad_size6;
            X_range = [X_new-pad_size6 : X_new+pad_size6];
      
            Y_range = [Y_new-pad_size6 : Y_new+pad_size6]; 

            img_data=im_extend(X_range,Y_range,:);
            
            img_data=imresize(img_data,[31 31],'bicubic');
            
            TEST_DATA(:,:,:,count)=img_data;
            
        
        end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%   write test.hdf5 %%%%%%%%%%%%%%%%%%%%%%%%%%%
TEST_INDEX=test_index(:,order2);
TEST_DATA=TEST_DATA(:,:,:,order2);
TEST_DATA=permute(TEST_DATA,[2 1 3 4]);
TEST_LABEL=TEST_LABEL(:,order2);
for k=1:count_batch
    count=length(TEST_LABEL)/count_batch;
    data=TEST_DATA(:,:,:,(k-1)*count+1:k*count);
    label=TEST_LABEL(:,(k-1)*count+1:k*count);
    chunksz = 100;
    created_flag = false;
    totalct = 0;
    savepath_test_temp=strcat(savepath_test13,'test',num2str(k),'.h5');
    for batchno = 1:ceil(count/chunksz)
        last_read=(batchno-1)*chunksz;
        if batchno*chunksz>count
            batchdata = data(:,:,:,last_read+1:end); 
            batchlabs = label(:,last_read+1:end); 
        else
            batchdata = data(:,:,:,last_read+1:last_read+chunksz); 
            batchlabs = label(:,last_read+1:last_read+chunksz);
        end

        startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,totalct+1]);
        curr_dat_sz = store2hdf5(savepath_test_temp, batchdata, batchlabs, ~created_flag, startloc, chunksz); 
        created_flag = true;
        totalct = curr_dat_sz(end);
    end
    h5disp(savepath_test_temp);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

save HSI13/data.mat TRAIN_INDEX TEST_INDEX TEST_LABEL TRAIN_LABEL train_index test_index train_label test_label











savepath_train15 = 'HSI15/train.h5';
savepath_test15  = 'HSI15/';


pad_size7=7;
input_size7=2*pad_size7+1;
%% number of training samples per class

%per_class_num=[1,15,9,3,5,8,1,5,1,10,24,6,3,13,4,1];                 % 1% sampling. test: 10140=20*507' train: 109 
%per_class_num=[3,72,42,12,25,37,2,24,1,49,122,30,11,64,20,5];        % 5%  sampling. test: 9730=10*973; train: 519
%per_class_num=[5,100,59,17,34,52,3,34,3,69,172,42,15,89,28,7];      % 7% sampling. test: 9520=20*476; train: 729
per_class_num=[5,143,83,24,49,73,3,48,2,98,245,60,21,126,39,10];     % 10% sampling. test: 9220=20*461; train: 1029
%per_class_num=[7,214,124,36,73,110,5,72,3,146,368,89,31,189,58,14];   % 15% samling.  test: 8710=10*871; train: 1539
%per_class_num=[10,285,165,48,97,145,6,96,4,194,490,119,41,252,78,19];       % 20% sampling. test: 8200=50*164; train: 2049


%%%%%% extending the image %%%%%%%%
im_extend=padarray(im,[pad_size7,pad_size7],'symmetric');




%%%%%  generate training samples and labels%%%%%%%%%%%%%%%%%%%%%%
TRAIN_DATA=zeros(31,31,num_bands,length(train_label));
TRAIN_LABEL=train_label-1;
count=0;
    
        for j=1:length(train_index)
            count=count+1;

            img_data=[];
            X=mod(train_index(j),I_row);
            Y=ceil(train_index(j)/I_row);
            if X==0
               X=I_row;
            end
            if Y==0
              Y=I_line;
            end
            X_new = X+pad_size7;
            Y_new = Y+pad_size7;
            X_range = [X_new-pad_size7 : X_new+pad_size7];
      
            Y_range = [Y_new-pad_size7 : Y_new+pad_size7]; 

            img_data=im_extend(X_range,Y_range,:);
            
            img_data=imresize(img_data,[31 31],'bicubic');
            
            TRAIN_DATA(:,:,:,count)=img_data;
            
        
        end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%    write train.hdf5    %%%%%%%%%%%%%
TRAIN_INDEX=train_index(:,order1);
TRAIN_DATA=TRAIN_DATA(:,:,:,order1);
TRAIN_DATA=permute(TRAIN_DATA,[2 1 3 4]);
TRAIN_LABEL=TRAIN_LABEL(:,order1);
data=TRAIN_DATA;
label=TRAIN_LABEL;
chunksz = 100;
created_flag = false;
totalct = 0;

for batchno = 1:ceil(count/chunksz)
    last_read=(batchno-1)*chunksz;
    if batchno*chunksz>count
        batchdata = data(:,:,:,last_read+1:end); 
        batchlabs = label(:,last_read+1:end); 
    else
        batchdata = data(:,:,:,last_read+1:last_read+chunksz); 
        batchlabs = label(:,last_read+1:last_read+chunksz);
    end

    startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,totalct+1]);
    curr_dat_sz = store2hdf5(savepath_train15, batchdata, batchlabs, ~created_flag, startloc, chunksz); 
    created_flag = true;
    totalct = curr_dat_sz(end);
end
h5disp(savepath_train15);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%% generate testing samples and labels  %%%%%%%%%%%%%%%%%%
TEST_DATA=zeros(31,31,num_bands,length(test_label));
TEST_LABEL=test_label-1;
count=0;
   
        for j=1:length(test_index)
            count=count+1;

            img_data=[];
            X=mod(test_index(j),I_row);
            Y=ceil(test_index(j)/I_row);
            if X==0
               X=I_row;
            end
            if Y==0
              Y=I_line;
            end
            X_new = X+pad_size7;
            Y_new = Y+pad_size7;
            X_range = [X_new-pad_size7 : X_new+pad_size7];
      
            Y_range = [Y_new-pad_size7 : Y_new+pad_size7]; 

            img_data=im_extend(X_range,Y_range,:);
            
            img_data=imresize(img_data,[31 31],'bicubic');
            
            TEST_DATA(:,:,:,count)=img_data;
            
        
        end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%   write test.hdf5 %%%%%%%%%%%%%%%%%%%%%%%%%%%
TEST_INDEX=test_index(:,order2);
TEST_DATA=TEST_DATA(:,:,:,order2);
TEST_DATA=permute(TEST_DATA,[2 1 3 4]);
TEST_LABEL=TEST_LABEL(:,order2);
for k=1:count_batch
    count=length(TEST_LABEL)/count_batch;
    data=TEST_DATA(:,:,:,(k-1)*count+1:k*count);
    label=TEST_LABEL(:,(k-1)*count+1:k*count);
    chunksz = 100;
    created_flag = false;
    totalct = 0;
    savepath_test_temp=strcat(savepath_test15,'test',num2str(k),'.h5');
    for batchno = 1:ceil(count/chunksz)
        last_read=(batchno-1)*chunksz;
        if batchno*chunksz>count
            batchdata = data(:,:,:,last_read+1:end); 
            batchlabs = label(:,last_read+1:end); 
        else
            batchdata = data(:,:,:,last_read+1:last_read+chunksz); 
            batchlabs = label(:,last_read+1:last_read+chunksz);
        end

        startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,totalct+1]);
        curr_dat_sz = store2hdf5(savepath_test_temp, batchdata, batchlabs, ~created_flag, startloc, chunksz); 
        created_flag = true;
        totalct = curr_dat_sz(end);
    end
    h5disp(savepath_test_temp);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

save HSI15/data.mat TRAIN_INDEX TEST_INDEX TEST_LABEL TRAIN_LABEL train_index test_index train_label test_label









savepath_train21 = 'HSI21/train.h5';
savepath_test21  = 'HSI21/';


pad_size10=10;
input_size10=2*pad_size10+1;
%% number of training samples per class

%per_class_num=[1,15,9,3,5,8,1,5,1,10,24,6,3,13,4,1];                 % 1% sampling. test: 10140=20*507' train: 109 
%per_class_num=[3,72,42,12,25,37,2,24,1,49,122,30,11,64,20,5];        % 5%  sampling. test: 9730=10*973; train: 519
%per_class_num=[5,100,59,17,34,52,3,34,3,69,172,42,15,89,28,7];      % 7% sampling. test: 9520=20*476; train: 729
per_class_num=[5,143,83,24,49,73,3,48,2,98,245,60,21,126,39,10];     % 10% sampling. test: 9220=20*461; train: 1029
%per_class_num=[7,214,124,36,73,110,5,72,3,146,368,89,31,189,58,14];   % 15% samling.  test: 8710=10*871; train: 1539
%per_class_num=[10,285,165,48,97,145,6,96,4,194,490,119,41,252,78,19];       % 20% sampling. test: 8200=50*164; train: 2049


%%%%%% extending the image %%%%%%%%
im_extend=padarray(im,[pad_size10,pad_size10],'symmetric');




%%%%%  generate training samples and labels%%%%%%%%%%%%%%%%%%%%%%
TRAIN_DATA=zeros(31,31,num_bands,length(train_label));
TRAIN_LABEL=train_label-1;
count=0;
    
        for j=1:length(train_index)
            count=count+1;

            img_data=[];
            X=mod(train_index(j),I_row);
            Y=ceil(train_index(j)/I_row);
            if X==0
               X=I_row;
            end
            if Y==0
              Y=I_line;
            end
            X_new = X+pad_size10;
            Y_new = Y+pad_size10;
            X_range = [X_new-pad_size10 : X_new+pad_size10];
      
            Y_range = [Y_new-pad_size10 : Y_new+pad_size10]; 

            img_data=im_extend(X_range,Y_range,:);
            
            img_data=imresize(img_data,[31 31],'bicubic');
            
            TRAIN_DATA(:,:,:,count)=img_data;
            
        
        end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%    write train.hdf5    %%%%%%%%%%%%%
TRAIN_INDEX=train_index(:,order1);
TRAIN_DATA=TRAIN_DATA(:,:,:,order1);
TRAIN_DATA=permute(TRAIN_DATA,[2 1 3 4]);
TRAIN_LABEL=TRAIN_LABEL(:,order1);
data=TRAIN_DATA;
label=TRAIN_LABEL;
chunksz = 100;
created_flag = false;
totalct = 0;

for batchno = 1:ceil(count/chunksz)
    last_read=(batchno-1)*chunksz;
    if batchno*chunksz>count
        batchdata = data(:,:,:,last_read+1:end); 
        batchlabs = label(:,last_read+1:end); 
    else
        batchdata = data(:,:,:,last_read+1:last_read+chunksz); 
        batchlabs = label(:,last_read+1:last_read+chunksz);
    end

    startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,totalct+1]);
    curr_dat_sz = store2hdf5(savepath_train21, batchdata, batchlabs, ~created_flag, startloc, chunksz); 
    created_flag = true;
    totalct = curr_dat_sz(end);
end
h5disp(savepath_train21);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%% generate testing samples and labels  %%%%%%%%%%%%%%%%%%
TEST_DATA=zeros(31,31,num_bands,length(test_label));
TEST_LABEL=test_label-1;
count=0;
   
        for j=1:length(test_index)
            count=count+1;

            img_data=[];
            X=mod(test_index(j),I_row);
            Y=ceil(test_index(j)/I_row);
            if X==0
               X=I_row;
            end
            if Y==0
              Y=I_line;
            end
            X_new = X+pad_size10;
            Y_new = Y+pad_size10;
            X_range = [X_new-pad_size10 : X_new+pad_size10];
      
            Y_range = [Y_new-pad_size10 : Y_new+pad_size10]; 

            img_data=im_extend(X_range,Y_range,:);
            
            img_data=imresize(img_data,[31 31],'bicubic');
            
            TEST_DATA(:,:,:,count)=img_data;
            
        
        end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%   write test.hdf5 %%%%%%%%%%%%%%%%%%%%%%%%%%%
TEST_INDEX=test_index(:,order2);
TEST_DATA=TEST_DATA(:,:,:,order2);
TEST_DATA=permute(TEST_DATA,[2 1 3 4]);
TEST_LABEL=TEST_LABEL(:,order2);
for k=1:count_batch
    count=length(TEST_LABEL)/count_batch;
    data=TEST_DATA(:,:,:,(k-1)*count+1:k*count);
    label=TEST_LABEL(:,(k-1)*count+1:k*count);
    chunksz = 100;
    created_flag = false;
    totalct = 0;
    savepath_test_temp=strcat(savepath_test21,'test',num2str(k),'.h5');
    for batchno = 1:ceil(count/chunksz)
        last_read=(batchno-1)*chunksz;
        if batchno*chunksz>count
            batchdata = data(:,:,:,last_read+1:end); 
            batchlabs = label(:,last_read+1:end); 
        else
            batchdata = data(:,:,:,last_read+1:last_read+chunksz); 
            batchlabs = label(:,last_read+1:last_read+chunksz);
        end

        startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,totalct+1]);
        curr_dat_sz = store2hdf5(savepath_test_temp, batchdata, batchlabs, ~created_flag, startloc, chunksz); 
        created_flag = true;
        totalct = curr_dat_sz(end);
    end
    h5disp(savepath_test_temp);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

save HSI21/data.mat TRAIN_INDEX TEST_INDEX TEST_LABEL TRAIN_LABEL train_index test_index train_label test_label













savepath_train25 = 'HSI25/train.h5';
savepath_test25  = 'HSI25/';


pad_size12=12;
input_size12=2*pad_size12+1;
%% number of training samples per class

%per_class_num=[1,15,9,3,5,8,1,5,1,10,24,6,3,13,4,1];                 % 1% sampling. test: 10140=20*507' train: 109 
%per_class_num=[3,72,42,12,25,37,2,24,1,49,122,30,11,64,20,5];        % 5%  sampling. test: 9730=10*973; train: 519
%per_class_num=[5,100,59,17,34,52,3,34,3,69,172,42,15,89,28,7];      % 7% sampling. test: 9520=20*476; train: 729
per_class_num=[5,143,83,24,49,73,3,48,2,98,245,60,21,126,39,10];     % 10% sampling. test: 9220=20*461; train: 1029
%per_class_num=[7,214,124,36,73,110,5,72,3,146,368,89,31,189,58,14];   % 15% samling.  test: 8710=10*871; train: 1539
%per_class_num=[10,285,165,48,97,145,6,96,4,194,490,119,41,252,78,19];       % 20% sampling. test: 8200=50*164; train: 2049


%%%%%% extending the image %%%%%%%%
im_extend=padarray(im,[pad_size12,pad_size12],'symmetric');




%%%%%  generate training samples and labels%%%%%%%%%%%%%%%%%%%%%%
TRAIN_DATA=zeros(31,31,num_bands,length(train_label));
TRAIN_LABEL=train_label-1;
count=0;
    
        for j=1:length(train_index)
            count=count+1;

            img_data=[];
            X=mod(train_index(j),I_row);
            Y=ceil(train_index(j)/I_row);
            if X==0
               X=I_row;
            end
            if Y==0
              Y=I_line;
            end
            X_new = X+pad_size12;
            Y_new = Y+pad_size12;
            X_range = [X_new-pad_size12 : X_new+pad_size12];
      
            Y_range = [Y_new-pad_size12 : Y_new+pad_size12]; 

            img_data=im_extend(X_range,Y_range,:);
            
            img_data=imresize(img_data,[31 31],'bicubic');
            
            TRAIN_DATA(:,:,:,count)=img_data;
            
        
        end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%    write train.hdf5    %%%%%%%%%%%%%
TRAIN_INDEX=train_index(:,order1);
TRAIN_DATA=TRAIN_DATA(:,:,:,order1);
TRAIN_DATA=permute(TRAIN_DATA,[2 1 3 4]);
TRAIN_LABEL=TRAIN_LABEL(:,order1);
data=TRAIN_DATA;
label=TRAIN_LABEL;
chunksz = 100;
created_flag = false;
totalct = 0;

for batchno = 1:ceil(count/chunksz)
    last_read=(batchno-1)*chunksz;
    if batchno*chunksz>count
        batchdata = data(:,:,:,last_read+1:end); 
        batchlabs = label(:,last_read+1:end); 
    else
        batchdata = data(:,:,:,last_read+1:last_read+chunksz); 
        batchlabs = label(:,last_read+1:last_read+chunksz);
    end

    startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,totalct+1]);
    curr_dat_sz = store2hdf5(savepath_train25, batchdata, batchlabs, ~created_flag, startloc, chunksz); 
    created_flag = true;
    totalct = curr_dat_sz(end);
end
h5disp(savepath_train25);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%% generate testing samples and labels  %%%%%%%%%%%%%%%%%%
TEST_DATA=zeros(31,31,num_bands,length(test_label));
TEST_LABEL=test_label-1;
count=0;
   
        for j=1:length(test_index)
            count=count+1;

            img_data=[];
            X=mod(test_index(j),I_row);
            Y=ceil(test_index(j)/I_row);
            if X==0
               X=I_row;
            end
            if Y==0
              Y=I_line;
            end
            X_new = X+pad_size12;
            Y_new = Y+pad_size12;
            X_range = [X_new-pad_size12 : X_new+pad_size12];
      
            Y_range = [Y_new-pad_size12 : Y_new+pad_size12]; 

            img_data=im_extend(X_range,Y_range,:);
            
            img_data=imresize(img_data,[31 31],'bicubic');
            
            TEST_DATA(:,:,:,count)=img_data;
            
        
        end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%   write test.hdf5 %%%%%%%%%%%%%%%%%%%%%%%%%%%
TEST_INDEX=test_index(:,order2);
TEST_DATA=TEST_DATA(:,:,:,order2);
TEST_DATA=permute(TEST_DATA,[2 1 3 4]);
TEST_LABEL=TEST_LABEL(:,order2);
for k=1:count_batch
    count=length(TEST_LABEL)/count_batch;
    data=TEST_DATA(:,:,:,(k-1)*count+1:k*count);
    label=TEST_LABEL(:,(k-1)*count+1:k*count);
    chunksz = 100;
    created_flag = false;
    totalct = 0;
    savepath_test_temp=strcat(savepath_test25,'test',num2str(k),'.h5');
    for batchno = 1:ceil(count/chunksz)
        last_read=(batchno-1)*chunksz;
        if batchno*chunksz>count
            batchdata = data(:,:,:,last_read+1:end); 
            batchlabs = label(:,last_read+1:end); 
        else
            batchdata = data(:,:,:,last_read+1:last_read+chunksz); 
            batchlabs = label(:,last_read+1:last_read+chunksz);
        end

        startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,totalct+1]);
        curr_dat_sz = store2hdf5(savepath_test_temp, batchdata, batchlabs, ~created_flag, startloc, chunksz); 
        created_flag = true;
        totalct = curr_dat_sz(end);
    end
    h5disp(savepath_test_temp);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

save HSI25/data.mat TRAIN_INDEX TEST_INDEX TEST_LABEL TRAIN_LABEL train_index test_index train_label test_label











savepath_train31 = 'HSI31/train.h5';
savepath_test31  = 'HSI31/';


pad_size15=15;
input_size15=2*pad_size15+1;
%% number of training samples per class

%per_class_num=[1,15,9,3,5,8,1,5,1,10,24,6,3,13,4,1];                 % 1% sampling. test: 10140=20*507' train: 109 
%per_class_num=[3,72,42,12,25,37,2,24,1,49,122,30,11,64,20,5];        % 5%  sampling. test: 9730=10*973; train: 519
%per_class_num=[5,100,59,17,34,52,3,34,3,69,172,42,15,89,28,7];      % 7% sampling. test: 9520=20*476; train: 729
per_class_num=[5,143,83,24,49,73,3,48,2,98,245,60,21,126,39,10];     % 10% sampling. test: 9220=20*461; train: 1029
%per_class_num=[7,214,124,36,73,110,5,72,3,146,368,89,31,189,58,14];   % 15% samling.  test: 8710=10*871; train: 1539
%per_class_num=[10,285,165,48,97,145,6,96,4,194,490,119,41,252,78,19];       % 20% sampling. test: 8200=50*164; train: 2049


%%%%%% extending the image %%%%%%%%
im_extend=padarray(im,[pad_size15,pad_size15],'symmetric');




%%%%%  generate training samples and labels%%%%%%%%%%%%%%%%%%%%%%
TRAIN_DATA=zeros(31,31,num_bands,length(train_label));
TRAIN_LABEL=train_label-1;
count=0;
    
        for j=1:length(train_index)
            count=count+1;

            img_data=[];
            X=mod(train_index(j),I_row);
            Y=ceil(train_index(j)/I_row);
            if X==0
               X=I_row;
            end
            if Y==0
              Y=I_line;
            end
            X_new = X+pad_size15;
            Y_new = Y+pad_size15;
            X_range = [X_new-pad_size15 : X_new+pad_size15];
      
            Y_range = [Y_new-pad_size15 : Y_new+pad_size15]; 

            img_data=im_extend(X_range,Y_range,:);
            
            
            TRAIN_DATA(:,:,:,count)=img_data;
            
        
        end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%    write train.hdf5    %%%%%%%%%%%%%
TRAIN_INDEX=train_index(:,order1);
TRAIN_DATA=TRAIN_DATA(:,:,:,order1);
TRAIN_DATA=permute(TRAIN_DATA,[2 1 3 4]);
TRAIN_LABEL=TRAIN_LABEL(:,order1);
data=TRAIN_DATA;
label=TRAIN_LABEL;
chunksz = 100;
created_flag = false;
totalct = 0;

for batchno = 1:ceil(count/chunksz)
    last_read=(batchno-1)*chunksz;
    if batchno*chunksz>count
        batchdata = data(:,:,:,last_read+1:end); 
        batchlabs = label(:,last_read+1:end); 
    else
        batchdata = data(:,:,:,last_read+1:last_read+chunksz); 
        batchlabs = label(:,last_read+1:last_read+chunksz);
    end

    startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,totalct+1]);
    curr_dat_sz = store2hdf5(savepath_train31, batchdata, batchlabs, ~created_flag, startloc, chunksz); 
    created_flag = true;
    totalct = curr_dat_sz(end);
end
h5disp(savepath_train31);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%% generate testing samples and labels  %%%%%%%%%%%%%%%%%%
TEST_DATA=zeros(31,31,num_bands,length(test_label));
TEST_LABEL=test_label-1;
count=0;
   
        for j=1:length(test_index)
            count=count+1;

            img_data=[];
            X=mod(test_index(j),I_row);
            Y=ceil(test_index(j)/I_row);
            if X==0
               X=I_row;
            end
            if Y==0
              Y=I_line;
            end
            X_new = X+pad_size15;
            Y_new = Y+pad_size15;
            X_range = [X_new-pad_size15 : X_new+pad_size15];
      
            Y_range = [Y_new-pad_size15 : Y_new+pad_size15]; 

            img_data=im_extend(X_range,Y_range,:);
            
            
            TEST_DATA(:,:,:,count)=img_data;
            
        
        end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%   write test.hdf5 %%%%%%%%%%%%%%%%%%%%%%%%%%%
TEST_INDEX=test_index(:,order2);
TEST_DATA=TEST_DATA(:,:,:,order2);
TEST_DATA=permute(TEST_DATA,[2 1 3 4]);
TEST_LABEL=TEST_LABEL(:,order2);
for k=1:count_batch
    count=length(TEST_LABEL)/count_batch;
    data=TEST_DATA(:,:,:,(k-1)*count+1:k*count);
    label=TEST_LABEL(:,(k-1)*count+1:k*count);
    chunksz = 100;
    created_flag = false;
    totalct = 0;
    savepath_test_temp=strcat(savepath_test31,'test',num2str(k),'.h5');
    for batchno = 1:ceil(count/chunksz)
        last_read=(batchno-1)*chunksz;
        if batchno*chunksz>count
            batchdata = data(:,:,:,last_read+1:end); 
            batchlabs = label(:,last_read+1:end); 
        else
            batchdata = data(:,:,:,last_read+1:last_read+chunksz); 
            batchlabs = label(:,last_read+1:last_read+chunksz);
        end

        startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,totalct+1]);
        curr_dat_sz = store2hdf5(savepath_test_temp, batchdata, batchlabs, ~created_flag, startloc, chunksz); 
        created_flag = true;
        totalct = curr_dat_sz(end);
    end
    h5disp(savepath_test_temp);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

save HSI31/data.mat TRAIN_INDEX TEST_INDEX TEST_LABEL TRAIN_LABEL train_index test_index train_label test_label
