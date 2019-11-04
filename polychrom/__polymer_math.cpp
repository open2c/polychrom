#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <ctime>
#include <omp.h>

using namespace std;

struct point{
    double x,y,z;
    point operator + (const point &p) const {
        return (point) {x+p.x, y+p.y, z+p.z};
    }
    point operator - (const point &p) const {
        return (point) {x-p.x, y-p.y, z-p.z};
    }
/* cross product */
    point operator * (const point &p) const {
        return (point) {y*p.z - z*p.y,
                        z*p.x - x*p.z,
                        x*p.y - y*p.x};
    }
    point operator * (const double &d) const {
        return (point) {d*x, d*y, d*z};
    }

    point operator / (const double &d) const {
        return (point) {x/d, y/d, z/d};
    }
};


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

inline double sqr(double x){
    return x*x;
}

inline double dotProduct(point a, point b){
    return a.x*b.x+a.y*b.y+a.z*b.z;
}

// inline double dist1(int i,int j){
//     return sqrt(dotProduct((position1[i]-position1[j]),(position1[i]-position1[j])));
// }

// inline double dist2(int i,int j){
//     return sqrt(dotProduct((position2[i]-position2[j]),(position2[i]-position2[j])));
// }

inline double dist(point a,point b){
    return sqrt( (a.x-b.x)*(a.x-b.x)
                +(a.y-b.y)*(a.y-b.y)
                +(a.z-b.z)*(a.z-b.z));
}

// inline double dist(int i,int j){
//     return sqrt(dotProduct((position[i]-position[j]),(position[i]-position[j])));
// }


int intersect(point t1,point t2,point t3,point r1,point r2) {
    point A,B,C,D,n;

    double det,t,u,v,c1,d1,d2,d3;
    B = t2 - t1;
    C = t3 - t1;
    D = r2 - t1;
    A = r2 - r1;

    d1 = (B.y*C.z-C.y*B.z);
    d2 = (B.x*C.z-B.z*C.x);
    d3 = (B.x*C.y-C.x*B.y);
    det = A.x*d1-A.y*d2+A.z*d3;
    if (det == 0) return 0;
    if (det >0){
        t = D.x*d1-D.y*d2+D.z*d3;
        if (t<0 || t>det) return 0;
        u = A.x*(D.y*C.z-C.y*D.z)-A.y*(D.x*C.z-D.z*C.x)+A.z*(D.x*C.y-C.x*D.y);
        if (u<0 || u>det) return 0;
        v = A.x*(B.y*D.z-D.y*B.z)-A.y*(B.x*D.z-B.z*D.x)+A.z*(B.x*D.y-D.x*B.y);
        if (v<0 || v>det || (u+v)>det) return 0;
        //printf("\n%lf,%lf,%lf, ",t/det,u/det,v/det);
        n = B*C;
        c1 = dotProduct(r1-t1,n);
        if (c1>0) return 1;
        else return -1;
    }
    else {
        t = D.x*d1-D.y*d2+D.z*d3;
        if (t>0 || t<det) return 0;
        u = A.x*(D.y*C.z-C.y*D.z)-A.y*(D.x*C.z-D.z*C.x)+A.z*(D.x*C.y-C.x*D.y);
        if (u>0 || u<det) return 0;
        v = A.x*(B.y*D.z-D.y*B.z)-A.y*(B.x*D.z-B.z*D.x)+A.z*(B.x*D.y-D.x*B.y);
        if (v>0 || v<det || (u+v)<det) return 0;
        //printf("\n%lf,%lf,%lf, ",t/det,u/det,v/det);
        n = B*C;
        c1 = dotProduct(r1-t1,n);
        if (c1>0) return 1;
        else return -1;
    }
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


void _mutualSimplifyCpp ( 
    double *datax1, double *datay1, double *dataz1, int N1,
    double *datax2, double *datay2, double *dataz2, int N2,
    long *ret
    )
 {
    int M = 0;
    int sum = 0;
    int t=0,s=0,k=0, k1;
    int turn=0;
    bool breakflag;

    vector <point> position1;
    vector <point> newposition1;
    vector <int> todelete1;

    vector <point> position2;
    vector <point> newposition2;
    vector <int> todelete2;

    int i;

    position1=vector<point>(N1);
    newposition1=vector<point>(N1);

    position2=vector<point>(N2);
    newposition2=vector<point>(N2);

    for (i=0;i<N1;i++)
    {
    position1[i].x = datax1[i] +  0.000000000000001*(rand()%1000);
    position1[i].y = datay1[i] +  0.00000000000000001*(rand()%1000);
    position1[i].z = dataz1[i] +  0.00000000000000001*(rand()%1000);
    }

    for (i=0;i<N2;i++)
    {
    position2[i].x = datax2[i]  +  0.000000000000001*(rand()%1000);
    position2[i].y = datay2[i]  +  0.0000000000000000001*(rand()%1000);
    position2[i].z  = dataz2[i] +  0.0000000000000000001*(rand()%1000);
    }

    todelete1 = vector <int> (N1);
    todelete2 = vector <int> (N2);

    for (i=0;i<N1;i++) todelete1[i] == -2;
    for (i=0;i<N2;i++) todelete2[i] == -2;

    for (int ttt = 0; ttt < 1; ttt++)
        {
        turn++;
        M=0;
        for (i=0;i<N1;i++) todelete1[i] = -2;
        for (i=0;i<N2;i++) todelete2[i] = -2;

        for (int j=1;j<N1-1;j++)  //going over all elements trying to delete
            {

            breakflag = false; //by default we delete thing
            for (k=0;k<N1;k++)  //going over all triangles to check
                {
                if (k < j-2 || k > j+1)
                    {
                    if (k < N1 - 1) k1 = k + 1;
                    else k1 = 0;
                    sum = intersect(position1[j-1],position1[j],position1[
                        j+1],position1[k],position1[k1]);
                    if (sum!=0)
                        {
                        //printf("intersection at %d,%d\n",j,k);
                        breakflag = true; //keeping thing
                        break;
                        }
                    }
                }

            if (breakflag == false)
            {
            for (k=0;k<N2;k++)  //going over all triangles to check
                {
                    if (k < N2 - 1) k1 = k + 1;
                    else k1 = 0;
                    sum = intersect(position1[j-1],position1[j],position1[
                        j+1],position2[k],position2[k1]);
                    if (sum!=0)
                        {
                        //printf("crossintersection at %d,%d\n",j,k);
                        breakflag = true; //keeping thing
                        break;
                        }
                }
             }

            if (breakflag ==false)
            {
            todelete1[M++] = j;
            position1[j] = (position1[j-1] + position1[j+1])* 0.5;
            //printf("%d will be deleted at %d\n",j,k);
            j++;
            //break;
            }
            }
        t = 0;//pointer for todelete
        s = 0;//pointer for newposition
        if (M==0)
            {
            break;
            }
        for (int j=0;j<N1;j++)
            {
            if (todelete1[t] == j)
                {
                t++;
                continue;
                }
            else
                {
                newposition1[s++] = position1[j];
                }
            }
        N1 = s;
        M = 0;
        t = 0;
        position1 = newposition1;
        }

    ret[0] = N1;
    ret[1] = N2;

    for (i=0;i<N1;i++)
    {
    datax1[i]  = position1[i].x;
    datay1[i]  = position1[i].y;
    dataz1[i]  = position1[i].z;
    }
    for (i=0;i<N2;i++)
    {
    datax2[i]  = position2[i].x;
    datay2[i]  = position2[i].y;
    dataz2[i]  = position2[i].z;
    }
}


int _simplifyCpp (double *datax, double *datay, double *dataz, int N) 
{
    int M = 0;
    int k1;
    int sum = 0;
    int t=0,s=0,k=0;
    int turn=0;
    bool breakflag;
    float maxdist;
    vector <point> position;
    vector <point> newposition;
    vector <int> todelete;
    int i;

    position=vector<point>(N);
    newposition=vector<point>(N);

    for (i=0;i<N;i++)
    {
        position[i].x = datax[i] + 0.000000000001*(rand()%1000);
        position[i].y = datay[i] + 0.00000000000001*(rand()%1000);
        position[i].z = dataz[i] + 0.0000000000001*(rand()%1000);
    }

    todelete = vector <int> (N);
    for (i=0;i<N;i++) todelete[i] == -2;
    for (int xxx = 0; xxx < 1000; xxx++)
        {
        maxdist = 0;
        for (i=0;i<N-1;i++)
        {
        if (dist(position[i],position[i+1]) > maxdist) {maxdist = dist(position[i],position[i+1]);}
        }
        turn++;
        M=0;
        for (i=0;i<N;i++) todelete[i] = -2;
        for (int j=1;j<N-1;j++)  //going over all elements trying to delete
            {
            breakflag = false; //by default we delete thing

            for (k=0;k<N;k++)  //going over all triangles to check
                {
                long double dd = dist(position[j],position[k]);
                if (dd  <  2 * maxdist)
                {

                if (k < j-2 || k > j+1)
                    {
                    if (k < N-1) k1 = k+1;
                    else k1 = 0;
                    sum = intersect(
                        position[j-1],position[j],position[j+1],
                        position[k],position[k1]);
                    if (sum!=0)
                        {
                        //printf("intersection at %d,%d\n",j,k);
                        breakflag = true; //keeping thing
                        break;
                        }
                    }
		        }
		else
		{
			k+= max(((int)((float)dd/(float)maxdist )- 3), 0);
		}
                }
            if (breakflag ==false)
            {
            todelete[M++] = j;
            position[j] = (position[j-1] + position[j+1])* 0.5;
            //printf("%d will be deleted at %d\n",j,k);
            j++;
            //break;
            }
            }
        t = 0;//pointer for todelete
        s = 0;//pointer for newposition
        if (M==0)
            {
            break;
            }
        for (int j=0;j<N;j++)
            {
            if (todelete[t] == j)
                {
                t++;
                continue;
                }
            else
                {
                newposition[s++] = position[j];
                }
            }
        M = 0;
        t = 0;
        position = newposition;
    }

    for (i=0;i<s;i++)
    {
        datax[i]  = position[i].x;
        datay[i]  = position[i].y;
        dataz[i]  = position[i].z;
    }
    return s;
}