#include <iostream>
#include <fstream>

#include<math.h>

#include"cbackprop.h"
using namespace std;

double  sigmoid(double a)
{

   return 1.0f / (1.0f + exp(float((-1.0f) * a)));
}

double  cuadrada(double a)
{

   return a*a;
}
double  valor_absoluto(double a)
{
   if(a>=0)return a;
   return -a;
}

double binario(double a)
{
    if(a>=0.5)return 1;
    return 0;
}

int main()
{
   ifstream file("data.csv");
   int filas,columnas;
   file>>filas>>columnas;

   double data[filas][columnas];
   double testData[filas][columnas-1];
   for(int i=0;i<filas;i++)
   {   string dato;int j;
       for(j=0;j<columnas-1;j++)
       {
            getline(file,dato,',');
            data[i][j]=std::atoi(dato.c_str());
            testData[i][j]=std::atoi(dato.c_str());
            cout<<dato<<" ";

       }
       getline(file,dato,'\n');
       data[i][j]=std::atoi(dato.c_str());
       cout<<dato<<endl;


   }

   int numLayers = 5;
   int *lSz=new int[numLayers]{columnas-1,5,7,2,1};
    //lSz = {3,3,3,1};
   double beta = 0.2, alpha = 0.01, thresh = 0.00001;
   long num_iter = 1000000;


     //double (*funciones[])(double)={sigmoid,cuadrada,valor_absoluto,sigmoid};
     double (*funciones[])(double)={sigmoid,sigmoid,sigmoid,sigmoid,sigmoid};

   CBackProp *bp = new CBackProp(numLayers, lSz, beta, alpha);//layers, size, ---

   //train
   for (long i=0; i < num_iter ; i++)
   {
       bp->bpgt(data[i%filas], &data[i%filas][columnas-1],funciones);
       double error=bp->mse(&data[i%filas][columnas-1]);
       //cout<<error<<endl;
       if(error < thresh)
           break; // mse < threshold - acabo entrenar
   }



   for (int i = 0 ; i < filas ; i++ )
   {
       bp->ffwd(testData[i],funciones);
       for(int j=0;j<columnas-1;j++)
       {
            cout << testData[i][j]<< "  ";
       }
       cout<< bp->Out(0) << endl;
   }
}




