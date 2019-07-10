#include <stdio.h>
#include <stdlib.h>


double *cross(double *v1, double *v2) {
    double *v1xv2 = new double[3];
    v1xv2[0]=-v1[2]*v2[1] + v1[1]*v2[2];
    v1xv2[1]=v1[2]*v2[0] - v1[0]*v2[2];
    v1xv2[2]=-v1[1]*v2[0] + v1[0]*v2[1];
    return v1xv2;
}

double *linearCombo(double *v1, double *v2, double s1, double s2) {
    double *c = new double[3];
    c[0]=s1*v1[0]+s2*v2[0];
    c[1]=s1*v1[1]+s2*v2[1];
    c[2]=s1*v1[2]+s2*v2[2];
    return c;
}

long int intersectValue(double *p1, double *v1, double *p2, double *v2) {
    int x=0;
    double *v2xp2 = cross(v2,p2), *v2xp1 = cross(v2,p1), *v2xv1 = cross(v2,v1);
    double *v1xp1 = cross(v1,p1), *v1xp2 = cross(v1,p2), *v1xv2 = cross(v1,v2);
    double t1 = (v2xp2[2]-v2xp1[2])/v2xv1[2];
    double t2 = (v1xp1[2]-v1xp2[2])/v1xv2[2];
    if(t1<0 || t1>1 || t2<0 || t2>1) {
        free(v2xp2);free(v2xp1);free(v2xv1);free(v1xp1);free(v1xp2);free(v1xv2);
        return 0;
    }
    else {
        if(v1xv2[2]>=0) x=1;
        else x=-1;
    }
    double *inter1 = linearCombo(p1,v1,1,t1), *inter2 = linearCombo(p2,v2,1,t2);
    double z1 = inter1[2];
    double z2 = inter2[2];

    free(v2xp2);free(v2xp1);free(v2xv1);free(v1xp1);free(v1xp2);free(v1xv2);free(inter1);free(inter2);
    if(z1>=z2) return x;
    else return -x;
}


long int _getLinkingNumberCpp(int M, double *olddata, int N) {
    double **data = new double*[N];
    long int i,j;
    for(i=0;i<N;i++) {
        data[i] = new double[3];
        data[i][0]=olddata[3*i];
        data[i][1]=olddata[3*i + 1];
        data[i][2]=olddata[3*i + 2];
    }

    long int L = 0;
    for(i=0;i<M;i++) {
        for(j=M;j<N;j++) {
            double *v1, *v2;
            if(i<M-1) v1 = linearCombo(data[i+1],data[i],1,-1);
            else v1 = linearCombo(data[0],data[M-1],1,-1);

            if(j<N-1) v2 = linearCombo(data[j+1],data[j],1,-1);
            else v2 = linearCombo(data[M],data[N-1],1,-1);
            L+=intersectValue(data[i],v1,data[j],v2);
            free(v1);free(v2);
        }
    }

    return L;
}
