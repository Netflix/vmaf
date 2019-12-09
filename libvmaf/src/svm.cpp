#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <float.h>
#include <string.h>
#include <stdarg.h>
#include <limits.h>
#include <locale.h>
#include "svm.h"
int libsvm_version = LIBSVM_VERSION;
typedef float Qfloat;
typedef signed char schar;
#ifndef min
template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
#endif
#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif
template <class T> static inline void swap(T& x, T& y) { T t=x; x=y; y=t; }
template <class S, class T> static inline void clone(T*& dst, S* src, int n)
{
    dst = new T[n];
    memcpy((void *)dst,(void *)src,sizeof(T)*n);
}
static inline double powi(double base, int times)
{
    double tmp = base, ret = 1.0;

    for(int t=times; t>0; t/=2)
    {
        if(t%2==1) ret*=tmp;
        tmp = tmp * tmp;
    }
    return ret;
}
#define INF HUGE_VAL
#define TAU 1e-12
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

//
// Kernel evaluation
//
// the static method k_function is for doing single kernel evaluation
// the constructor of Kernel prepares to calculate the l*l kernel matrix
// the member function get_Q is for getting one column from the Q Matrix
//
class QMatrix {
public:
    virtual Qfloat *get_Q(int column, int len) const = 0;
    virtual double *get_QD() const = 0;
    virtual void swap_index(int i, int j) const = 0;
    virtual ~QMatrix() {}
};

class Kernel: public QMatrix {
public:
    Kernel(int l, svm_node * const * x, const svm_parameter& param);
    virtual ~Kernel();

    static double k_function(const svm_node *x, const svm_node *y,
                 const svm_parameter& param);
    virtual Qfloat *get_Q(int column, int len) const = 0;
    virtual double *get_QD() const = 0;
    virtual void swap_index(int i, int j) const    // no so const...
    {
        swap(x[i],x[j]);
        if(x_square) swap(x_square[i],x_square[j]);
    }
protected:

    double (Kernel::*kernel_function)(int i, int j) const;

private:
    const svm_node **x;
    double *x_square;

    // svm_parameter
    const int kernel_type;
    const int degree;
    const double gamma;
    const double coef0;

    static double dot(const svm_node *px, const svm_node *py);
    double kernel_linear(int i, int j) const
    {
        return dot(x[i],x[j]);
    }
    double kernel_poly(int i, int j) const
    {
        return powi(gamma*dot(x[i],x[j])+coef0,degree);
    }
    double kernel_rbf(int i, int j) const
    {
        return exp(-gamma*(x_square[i]+x_square[j]-2*dot(x[i],x[j])));
    }
    double kernel_sigmoid(int i, int j) const
    {
        return tanh(gamma*dot(x[i],x[j])+coef0);
    }
    double kernel_precomputed(int i, int j) const
    {
        return x[i][(int)(x[j][0].value)].value;
    }
};

Kernel::Kernel(int l, svm_node * const * x_, const svm_parameter& param)
:kernel_type(param.kernel_type), degree(param.degree),
 gamma(param.gamma), coef0(param.coef0)
{
    switch(kernel_type)
    {
        case LINEAR:
            kernel_function = &Kernel::kernel_linear;
            break;
        case POLY:
            kernel_function = &Kernel::kernel_poly;
            break;
        case RBF:
            kernel_function = &Kernel::kernel_rbf;
            break;
        case SIGMOID:
            kernel_function = &Kernel::kernel_sigmoid;
            break;
        case PRECOMPUTED:
            kernel_function = &Kernel::kernel_precomputed;
            break;
    }

    clone(x,x_,l);

    if(kernel_type == RBF)
    {
        x_square = new double[l];
        for(int i=0;i<l;i++)
            x_square[i] = dot(x[i],x[i]);
    }
    else
        x_square = 0;
}

Kernel::~Kernel()
{
    delete[] x;
    delete[] x_square;
}

double Kernel::dot(const svm_node *px, const svm_node *py)
{
    double sum = 0;
    while(px->index != -1 && py->index != -1)
    {
        if(px->index == py->index)
        {
            sum += px->value * py->value;
            ++px;
            ++py;
        }
        else
        {
            if(px->index > py->index)
                ++py;
            else
                ++px;
        }            
    }
    return sum;
}

double Kernel::k_function(const svm_node *x, const svm_node *y,
              const svm_parameter& param)
{
    switch(param.kernel_type)
    {
        case LINEAR:
            return dot(x,y);
        case POLY:
            return powi(param.gamma*dot(x,y)+param.coef0,param.degree);
        case RBF:
        {
            double sum = 0;
            while(x->index != -1 && y->index !=-1)
            {
                if(x->index == y->index)
                {
                    double d = x->value - y->value;
                    sum += d*d;
                    ++x;
                    ++y;
                }
                else
                {
                    if(x->index > y->index)
                    {    
                        sum += y->value * y->value;
                        ++y;
                    }
                    else
                    {
                        sum += x->value * x->value;
                        ++x;
                    }
                }
            }

            while(x->index != -1)
            {
                sum += x->value * x->value;
                ++x;
            }

            while(y->index != -1)
            {
                sum += y->value * y->value;
                ++y;
            }
            
            return exp(-param.gamma*sum);
        }
        case SIGMOID:
            return tanh(param.gamma*dot(x,y)+param.coef0);
        case PRECOMPUTED:  //x: test (validation), y: SV
            return x[(int)(y->value)].value;
        default:
            return 0;  // Unreachable 
    }
}

double svm_predict_values(const svm_model *model, const svm_node *x, double* dec_values)
{
    int i;
    if(model->param.svm_type == ONE_CLASS ||
       model->param.svm_type == EPSILON_SVR ||
       model->param.svm_type == NU_SVR)
    {
        double *sv_coef = model->sv_coef[0];
        double sum = 0;
        for(i=0;i<model->l;i++)
            sum += sv_coef[i] * Kernel::k_function(x,model->SV[i],model->param);
        sum -= model->rho[0];
        *dec_values = sum;

        if(model->param.svm_type == ONE_CLASS)
            return (sum>0)?1:-1;
        else
            return sum;
    }
    else
    {
        int nr_class = model->nr_class;
        int l = model->l;
        
        double *kvalue = Malloc(double,l);
        for(i=0;i<l;i++)
            kvalue[i] = Kernel::k_function(x,model->SV[i],model->param);

        int *start = Malloc(int,nr_class);
        start[0] = 0;
        for(i=1;i<nr_class;i++)
            start[i] = start[i-1]+model->nSV[i-1];

        int *vote = Malloc(int,nr_class);
        for(i=0;i<nr_class;i++)
            vote[i] = 0;

        int p=0;
        for(i=0;i<nr_class;i++)
            for(int j=i+1;j<nr_class;j++)
            {
                double sum = 0;
                int si = start[i];
                int sj = start[j];
                int ci = model->nSV[i];
                int cj = model->nSV[j];
                
                int k;
                double *coef1 = model->sv_coef[j-1];
                double *coef2 = model->sv_coef[i];
                for(k=0;k<ci;k++)
                    sum += coef1[si+k] * kvalue[si+k];
                for(k=0;k<cj;k++)
                    sum += coef2[sj+k] * kvalue[sj+k];
                sum -= model->rho[p];
                dec_values[p] = sum;

                if(dec_values[p] > 0)
                    ++vote[i];
                else
                    ++vote[j];
                p++;
            }

        int vote_max_idx = 0;
        for(i=1;i<nr_class;i++)
            if(vote[i] > vote[vote_max_idx])
                vote_max_idx = i;

        free(kvalue);
        free(start);
        free(vote);
        return model->label[vote_max_idx];
    }
}

double svm_predict(const svm_model *model, const svm_node *x)
{
    int nr_class = model->nr_class;
    double *dec_values;
    if(model->param.svm_type == ONE_CLASS ||
       model->param.svm_type == EPSILON_SVR ||
       model->param.svm_type == NU_SVR)
        dec_values = Malloc(double, 1);
    else 
        dec_values = Malloc(double, nr_class*(nr_class-1)/2);
    double pred_result = svm_predict_values(model, x, dec_values);
    free(dec_values);
    return pred_result;
}

static const char *svm_type_table[] =
{
    "c_svc","nu_svc","one_class","epsilon_svr","nu_svr",NULL
};

static const char *kernel_type_table[]=
{
    "linear","polynomial","rbf","sigmoid","precomputed",NULL
};

static char* readline(FILE *input, char* line, int max_line_len)
{
    int len;

    if(fgets(line,max_line_len,input) == NULL)
        return NULL;

    while(strrchr(line,'\n') == NULL)
    {
        max_line_len *= 2;
        line = (char *) realloc(line,max_line_len);
        len = (int) strlen(line);
        if(fgets(line+len,max_line_len-len,input) == NULL)
            break;
    }
    return line;
}

//
// FSCANF helps to handle fscanf failures.
// Its do-while block avoids the ambiguity when
// if (...)
//    FSCANF();
// is used
//
#define FSCANF(_stream, _format, _var) do{ if (fscanf(_stream, _format, _var) != 1) return false; }while(0)
bool read_model_header(FILE *fp, svm_model* model)
{
    svm_parameter& param = model->param;
    char cmd[81];
    while(1)
    {
        FSCANF(fp,"%80s",cmd);

        if(strcmp(cmd,"svm_type")==0)
        {
            FSCANF(fp,"%80s",cmd);
            int i;
            for(i=0;svm_type_table[i];i++)
            {
                if(strcmp(svm_type_table[i],cmd)==0)
                {
                    param.svm_type=i;
                    break;
                }
            }
            if(svm_type_table[i] == NULL)
            {
                fprintf(stderr,"unknown svm type.\n");
                return false;
            }
        }
        else if(strcmp(cmd,"kernel_type")==0)
        {        
            FSCANF(fp,"%80s",cmd);
            int i;
            for(i=0;kernel_type_table[i];i++)
            {
                if(strcmp(kernel_type_table[i],cmd)==0)
                {
                    param.kernel_type=i;
                    break;
                }
            }
            if(kernel_type_table[i] == NULL)
            {
                fprintf(stderr,"unknown kernel function.\n");    
                return false;
            }
        }
        else if(strcmp(cmd,"degree")==0)
            FSCANF(fp,"%d",&param.degree);
        else if(strcmp(cmd,"gamma")==0)
            FSCANF(fp,"%lf",&param.gamma);
        else if(strcmp(cmd,"coef0")==0)
            FSCANF(fp,"%lf",&param.coef0);
        else if(strcmp(cmd,"nr_class")==0)
            FSCANF(fp,"%d",&model->nr_class);
        else if(strcmp(cmd,"total_sv")==0)
            FSCANF(fp,"%d",&model->l);
        else if(strcmp(cmd,"rho")==0)
        {
            int n = model->nr_class * (model->nr_class-1)/2;
            model->rho = Malloc(double,n);
            for(int i=0;i<n;i++)
                FSCANF(fp,"%lf",&model->rho[i]);
        }
        else if(strcmp(cmd,"label")==0)
        {
            int n = model->nr_class;
            model->label = Malloc(int,n);
            for(int i=0;i<n;i++)
                FSCANF(fp,"%d",&model->label[i]);
        }
        else if(strcmp(cmd,"probA")==0)
        {
            int n = model->nr_class * (model->nr_class-1)/2;
            model->probA = Malloc(double,n);
            for(int i=0;i<n;i++)
                FSCANF(fp,"%lf",&model->probA[i]);
        }
        else if(strcmp(cmd,"probB")==0)
        {
            int n = model->nr_class * (model->nr_class-1)/2;
            model->probB = Malloc(double,n);
            for(int i=0;i<n;i++)
                FSCANF(fp,"%lf",&model->probB[i]);
        }
        else if(strcmp(cmd,"nr_sv")==0)
        {
            int n = model->nr_class;
            model->nSV = Malloc(int,n);
            for(int i=0;i<n;i++)
                FSCANF(fp,"%d",&model->nSV[i]);
        }
        else if(strcmp(cmd,"SV")==0)
        {
            while(1)
            {
                int c = getc(fp);
                if(c==EOF || c=='\n') break;
            }
            break;
        }
        else
        {
            fprintf(stderr,"unknown text in model file: [%s]\n",cmd);
            return false;
        }
    }

    return true;

}

svm_model *svm_load_model(const char *model_file_name)
{
    FILE *fp = fopen(model_file_name,"rb");
    if(fp==NULL) return NULL;

    char *old_locale = strdup(setlocale(LC_ALL, NULL));
    setlocale(LC_ALL, "C");

    // read parameters

    svm_model *model = Malloc(svm_model,1);
    model->rho = NULL;
    model->probA = NULL;
    model->probB = NULL;
    model->sv_indices = NULL;
    model->label = NULL;
    model->nSV = NULL;
    
    // read header
    if (!read_model_header(fp, model))
    {
        fprintf(stderr, "ERROR: fscanf failed to read model\n");
        setlocale(LC_ALL, old_locale);
        free(old_locale);
        free(model->rho);
        free(model->label);
        free(model->nSV);
        free(model);
        return NULL;
    }
    
    // read sv_coef and SV

    int elements = 0;
    long pos = ftell(fp);

    int max_line_len = 1024;
    char* line = Malloc(char,max_line_len);
    char *p,*endptr,*idx,*val;

    while(readline(fp, line, max_line_len)!=NULL)
    {
        p = strtok(line,":");
        while(1)
        {
            p = strtok(NULL,":");
            if(p == NULL)
                break;
            ++elements;
        }
    }
    elements += model->l;

    fseek(fp,pos,SEEK_SET);

    int m = model->nr_class - 1;
    int l = model->l;
    model->sv_coef = Malloc(double *,m);
    int i;
    for(i=0;i<m;i++)
        model->sv_coef[i] = Malloc(double,l);
    model->SV = Malloc(svm_node*,l);
    svm_node *x_space = NULL;
    if(l>0) x_space = Malloc(svm_node,elements);

    int j=0;
    for(i=0;i<l;i++)
    {
        readline(fp, line, max_line_len);
        model->SV[i] = &x_space[j];

        p = strtok(line, " \t");
        model->sv_coef[0][i] = strtod(p,&endptr);
        for(int k=1;k<m;k++)
        {
            p = strtok(NULL, " \t");
            model->sv_coef[k][i] = strtod(p,&endptr);
        }

        while(1)
        {
            idx = strtok(NULL, ":");
            val = strtok(NULL, " \t");

            if(val == NULL)
                break;
            x_space[j].index = (int) strtol(idx,&endptr,10);
            x_space[j].value = strtod(val,&endptr);

            ++j;
        }
        x_space[j++].index = -1;
    }
    free(line);

    setlocale(LC_ALL, old_locale);
    free(old_locale);

    if (ferror(fp) != 0 || fclose(fp) != 0)
        return NULL;

    model->free_sv = 1;    // XXX
    return model;
}

void svm_free_model_content(svm_model* model_ptr)
{
    if(model_ptr->free_sv && model_ptr->l > 0 && model_ptr->SV != NULL)
        free((void *)(model_ptr->SV[0]));
    if(model_ptr->sv_coef)
    {
        for(int i=0;i<model_ptr->nr_class-1;i++)
            free(model_ptr->sv_coef[i]);
    }

    free(model_ptr->SV);
    model_ptr->SV = NULL;

    free(model_ptr->sv_coef);
    model_ptr->sv_coef = NULL;

    free(model_ptr->rho);
    model_ptr->rho = NULL;

    free(model_ptr->label);
    model_ptr->label= NULL;

    free(model_ptr->probA);
    model_ptr->probA = NULL;

    free(model_ptr->probB);
    model_ptr->probB= NULL;

    free(model_ptr->sv_indices);
    model_ptr->sv_indices = NULL;

    free(model_ptr->nSV);
    model_ptr->nSV = NULL;
}

void svm_free_and_destroy_model(struct svm_model** model_ptr_ptr)
{
	if(model_ptr_ptr != NULL && *model_ptr_ptr != NULL)
	{
		svm_free_model_content(*model_ptr_ptr);
		free(*model_ptr_ptr);
		*model_ptr_ptr = NULL;
	}
}
