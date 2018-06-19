#include "cbackprop.h"
#include<math.h>

//Inicializa y separa memoria
CBackProp::CBackProp(int nl,int *sz,double b,double a):beta(b),alpha(a)
{

// no se usa
// delta[0]
// weight[0]
// prevDwt[0]

//  capa 0 - input
//  capa 1 - primera capa oculta
//  cap n  - capa salida.
//  capa 0 - solo almacena inputs - no hay delta o peso en esta

    ///  num de capas y sus tamaños
    numl=nl;
    lsize=new int[numl];

    for(int i=0;i<numl;i++){
        lsize[i]=sz[i];
    }

    ///  separar memoria de salida de cada neurona
    out = new double*[numl];

    for(int i=0;i<numl;i++){
        out[i]=new double[lsize[i]];
    }

    ///  separar memoria para delta
    delta = new double*[numl];

    for(int i=1;i<numl;i++){
        delta[i]=new double[lsize[i]];
    }

    ///  separar memoria para los pesos
    weight = new double**[numl];

    for(int i=1;i<numl;i++){
        weight[i]=new double*[lsize[i]];
    }
    for(int i=1;i<numl;i++){
        for(int j=0;j<lsize[i];j++){
            weight[i][j]=new double[lsize[i-1]+1];
        }
    }

    ///  separar memoria para pesos anteriores
    prevDwt = new double**[numl];

    for(int i=1;i<numl;i++){
        prevDwt[i]=new double*[lsize[i]];

    }
    for(int i=1;i<numl;i++){
        for(int j=0;j<lsize[i];j++){
            prevDwt[i][j]=new double[lsize[i-1]+1];
        }
    }

    ///  asignar pesos aleatorios
    srand((unsigned)(time(nullptr)));
    for(int i=1;i<numl;i++)
        for(int j=0;j<lsize[i];j++)
            for(int k=0;k<lsize[i-1]+1;k++)
                weight[i][j][k]=(double)(rand())/(RAND_MAX/2) - 1;

    ///  inicializar pesos anteriores a 0 para primer iteracion
    for(int i=1;i<numl;i++)
        for(int j=0;j<lsize[i];j++)
            for(int k=0;k<lsize[i-1]+1;k++)
                prevDwt[i][j][k]=(double)0.0;
}

double CBackProp::mse(double *tgt)
{
    double mse=0;
    for(int i=0;i<lsize[numl-1];i++){
        mse+=(tgt[i]-out[numl-1][i])*(tgt[i]-out[numl-1][i]);
    }
    return mse/lsize[numl-1];
}

double CBackProp::Out(int i)const
{
    return this->out[numl-1][i];
}

/// avanzar un conjunto de input
void CBackProp::ffwd(double *in,double (*funcion[])(double))
{
        double sum;

 /// asignar contenido a la capa de entrada

        for(int i=0;i < lsize[0];i++)
                out[0][i]=in[i];

/// asingar valor de activacion-output- a cada neurona con funcion sigmoidea

        /// para cada capa
        for(int i=1;i < numl;i++){
                ///  para cada neurona en la capa actual
                for(int j=0;j < lsize[i];j++){
                        sum=0.0;
                        /// para la entrada de cada neurona en capa anterior
                        for(int k=0;k < lsize[i-1];k++){
                                /// aplicar pesos a los inputs y sumarlos en sum
                                sum+= out[i-1][k]*weight[i][j][k];
                        }
                        /// aplicar bias
                        sum+=weight[i][j][lsize[i-1]];
                        /// aplicar func sigmoidea
                        out[i][j]=funcion[i-1](sum);
                }
        }
}


void CBackProp::bpgt(double *in,double *tgt,double (*funcion[])(double))
{
    double sum;
    ffwd(in,funcion);
    for(int i=0;i < lsize[numl-1];i++){
       delta[numl-1][i]=out[numl-1][i]* (1-out[numl-1][i])*(tgt[i]-out[numl-1][i]);
    }
    for(int i=numl-2;i>0;i--){
        for(int j=0;j < lsize[i];j++){
            sum=0.0;
            for(int k=0;k < lsize[i+1];k++){
                    sum+=delta[i+1][k]*weight[i+1][k][j];
            }
            delta[i][j]=out[i][j]*(1-out[i][j])*sum;
        }
    }
    for(int i=1;i < numl;i++){
        for(int j=0;j < lsize[i];j++){
            for(int k=0;k < lsize[i-1];k++){
                    weight[i][j][k]+=alpha*prevDwt[i][j][k];
            }
            weight[i][j][lsize[i-1]]+=alpha*prevDwt[i][j][lsize[i-1]];
        }
    }
    for(int i=1;i < numl;i++){
        for(int j=0;j < lsize[i];j++){
            for(int k=0;k < lsize[i-1];k++){
                    prevDwt[i][j][k]=beta*delta[i][j]*out[i-1][k];
                    weight[i][j][k]+=prevDwt[i][j][k];
            }
            prevDwt[i][j][lsize[i-1]]=beta*delta[i][j];
            weight[i][j][lsize[i-1]]+=prevDwt[i][j][lsize[i-1]];
        }
    }
}
