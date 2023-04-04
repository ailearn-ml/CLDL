clear;
clc;
% Load the trainData and testData.

load('data/sample_data.mat');

SVPMLparams.AG = 2;
SVPMLparams.SVPneigh = 150;

SVPMLparams.mxitr = 200;
SVPMLparams.lambda = 1;
SVPMLparams.w_thresh = 0.01;
SVPMLparams.sp_thresh = 0.01;
SVPMLparams.c = 0.2;
SVPMLparams.numThreads = 32;

mkdir("./save/encoding/");

for outDim =5:5:30
    SVPMLparams.outDim = outDim;
    if exist("./save/encoding/"+"sample_data"+"_"+int2str(outDim)+".mat")
        disp(dataset_name+" "+outDim+" exist!");
        continue
    else
        disp("sample_data"+" "+outDim+" training!");

        Y = sparse(train_labels);

        cd label_encoding
        tic;
        [U] = SVP(Y, SVPMLparams);
        encoding_time = toc;
        fprintf('Encoding time of CLDL: %8.7f \n', encoding_time);
        encoding = U;
        cd ..

        save("./save/encoding/"+"sample_data"+"_"+int2str(outDim)+".mat","encoding","encoding_time")
    end
end
