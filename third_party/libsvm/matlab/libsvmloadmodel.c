#include "svm.h"
#include "mex.h"
#include "svm_model_matlab.h"

static void fake_answer(mxArray *plhs[])
{
    plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    struct svm_model *model;
    char *filename;
    const char *error_msg;
    int nr_feat;

    // check input
    if(nrhs != 2) {
        mexPrintf("Usage: model = libsvmloadmodel('filename', num_of_feature);\n");
        fake_answer(plhs);
        return;
    }
    if(!mxIsChar(prhs[0]) || mxGetM(prhs[0])!=1) {
        mexPrintf("filename should be given as string\n");
        fake_answer(plhs);
        return;
    }
    if(mxGetNumberOfElements(prhs[1])!=1) {
        mexPrintf("number of features should be given as scalar\n");
        fake_answer(plhs);
        return;
    }

    // get filename and number of features
    filename = mxArrayToString(prhs[0]);
    nr_feat = (int) *(mxGetPr(prhs[1]));

    // load model from file
    model = svm_load_model(filename);
    if (model == NULL) {
        mexPrintf("Error occured while reading from file.\n");
        fake_answer(plhs);
        mxFree(filename);
        return;
    }

    // convert MATLAB struct to C struct
    error_msg = model_to_matlab_structure(plhs, nr_feat, model);
    if(error_msg) {
        mexPrintf("Error: can't convert libsvm model to matrix structure: %s\n", error_msg);
        fake_answer(plhs);
    }

    // destroy model
    svm_free_and_destroy_model(&model);
    mxFree(filename);

    return;
}
