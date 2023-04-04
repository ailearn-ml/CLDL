function [U] = SVP(Y, SVPMLparams)
    addpath(genpath('prcpy'));
    w_thresh = SVPMLparams.w_thresh;
    spParam = SVPMLparams.sp_thresh;
    lambda = SVPMLparams.lambda;
    numThreads = SVPMLparams.numThreads;
    numNeighbors = SVPMLparams.SVPneigh;
    outDim = SVPMLparams.outDim;
    %Set the parameters for svp
    outDim = SVPMLparams.outDim;
    params.tol=1e-3;
    params.mxitr=SVPMLparams.mxitr;
    params.verbosity=1;
    %Y = data.Y;
    [nc, l] = size(Y);
    numNeighbors = min(numNeighbors, nc);
    outDim  = min(outDim, nc);
    
    normY = sqrt(sum((Y'.^2), 1)) + 1e-10;
    Y = bsxfun(@rdivide, Y', normY);
    Y = Y';
    
    Ytr = Y';
    dytr = Y';
    ds = Ytr';
    dsY = dytr';

    
    [Om, OmVal, neighborIdx] = findKNN_test(ds', numNeighbors, numThreads);
    
    %Setup svp for this dataset
    neighborIdx = neighborIdx';
    done = false;
    
    [I,J]=ind2sub([nc nc],Om(:));
    MOmega=sparse(I, J,OmVal(:), nc, nc);
    
    while(~done)
        try
            tic;
            [U, S, V]=lansvd(MOmega,outDim, 'L');
            Uinit=U*sqrt(S);
            Vinit=V*sqrt(S);
            [U, V]=WAltMin_asymm(Om(:), OmVal(:), params.mxitr, params.tol, Uinit, Vinit, numThreads);
            t = toc;
            done  = true;
        catch exception
            msgString = getReport(exception);
            disp(msgString);
            done = false;
        end
    end
end
