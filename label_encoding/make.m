mex COMPFLAGS="\$COMPFLAGS -std=c99" -I.. -largeArrayDims findKNN_test.cpp smat.cpp COMPFLAGS="/openmp $COMPFLAGS"
mex COMPFLAGS="\$COMPFLAGS -std=c99" -I.. -largeArrayDims updateU.cpp smat.cpp COMPFLAGS="/openmp $COMPFLAGS"
mex COMPFLAGS="\$COMPFLAGS -std=c99" -I.. -largeArrayDims updateV.cpp smat.cpp COMPFLAGS="/openmp $COMPFLAGS"
mex COMPFLAGS="\$COMPFLAGS -std=c99" -I.. -largeArrayDims compute_X_Omega.c smat.cpp COMPFLAGS="/openmp $COMPFLAGS" 