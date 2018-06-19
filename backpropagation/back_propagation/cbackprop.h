#ifndef CBACKPROP_H
#define CBACKPROP_H


#include<time.h>
#include<stdlib.h>

class CBackProp{

//      salida de cada neurona
        double **out;

//      valor de error delta para cada neurona
        double **delta;

//      array 3-D guardar pesos de cada neurona
        double ***weight;

//      num de capas incluyendo la capa de entrada
        int numl;

//      array de num de elementos para almacenar el tamaño de cada capa
        int *lsize;

//      learning rate
        double beta;

//      momentum
        double alpha;

//      almacenamiento para cambio de pesos hechos en la epoca anterior
        double ***prevDwt;


public:

        ~CBackProp();

//      inicializar y separar memoria
        CBackProp(int nl,int *sz,double b,double a);

//      error backpropagado para un conjunto de entradas
        void bpgt(double *in, double *tgt ,double (*funcion[])(double));

//      activacion forward  para un conjunto de entradas
        void ffwd(double *in,double (*funcion[])(double));

//      retorna error medio cuadratico
        double mse(double *tgt);

//      retorna la i-esima salida de la red
        double Out(int i) const;
};


#endif // CBACKPROP_H
