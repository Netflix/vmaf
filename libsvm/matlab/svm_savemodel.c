#include "../svm.h"
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
    int status;

    /* check input */
    if(nrhs != 2) {
        mexPrintf("Usage: svm_savemodel(model, 'filename');\n");
        fake_answer(plhs);
        return;
    }
    if(!mxIsStruct(prhs[0])) {
        mexPrintf("model file should be a struct array\n");
        fake_answer(plhs);
        return;
    }
    if(!mxIsChar(prhs[1]) || mxGetM(prhs[1])!=1) {
        mexPrintf("filename should be given as char(s)\n");
        fake_answer(plhs);
        return;
    }

    /* convert MATLAB struct to C struct */
    model = matlab_matrix_to_model(prhs[0], &error_msg);
    if(model == NULL) {
        mexPrintf("Error: can't read model: %s\n", error_msg);
        fake_answer(plhs);
        return;
    }

    /* get filename */
    filename = mxArrayToString(prhs[1]);

    /* save model to file */
    status = svm_save_model(filename,model);
    if (status != 0) {
        mexWarnMsgTxt("Error occured while writing to file.");
    }

    /* destroy model */
    svm_free_and_destroy_model(&model);
    mxFree(filename);

    /* return status value (0: success, -1: failure) */
    plhs[0] = mxCreateDoubleScalar(status);

    return;
}