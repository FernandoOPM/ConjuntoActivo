
import numpy as np
import scipy
from scipy.optimize import linprog




#utiliza programación lineal para encontrar un punto inicial factible
def encuentraFactible(Ai = 0, bi = 0, Ad = 0, bd = 0):
    if np.any(Ai) != 0:
        n = Ai.shape[1]
    else:
        n= Ad.shape[1]
    c= np.ones(n)
    if np.any(Ai) != 0  and np.any(Ad) != 0:
        x = linprog(c, A_eq=Ai, b_eq=bi, A_ub=Ad, b_ub=bd)["x"]
    elif np.any(Ai):
        x = linprog(c, A_eq=Ai, b_eq=bi)["x"]
    elif np.any(Ad):
        x = linprog(c, A_ub=Ad, b_ub=bd)["x"]
    return x




"""
Ai es la matriz re restricciones de igualdad,
bi es la matriz de restricciones de igualdad,
Ad es la matriz re restricciones de desigualdad,
bd es la matriz de restricciones de desigualdad,

Sea W la Matriz para las restricciones del espacio W_k
Wind es el vector que nos indica que indices de Ad pertenecen a W_k
"""
def defW(x, Ai = 0, bi = 0, Ad = 0, bd = 0, tol = 1e-10):
    W = np.copy(Ai)
    if np.any(Ad) != 0:
        n = Ad.shape[0]
        Wind = np.zeros(n)
        aux = np.matmul(Ad,x)
        for i in range(n):
            if abs(aux[i]-bd[i]) < tol:
                Wind[i] = 1 
                if np.all(W == 0):
                    W = Ad[i,:]    
                    #wj = np.array([i])
                else:
                    W = np.vstack((W, Ad[i,:]))
                    #wj = np.vstack((wj, [i]))
    return W,  Wind


#Metodo del Rango para resolver problema cuadratico con restricciones de igualdadpara obtener d
def metodoRango(G, A, c, b):
    Ginv = np.linalg.inv(G)
    if np.all(A == 0):
        x = -np.matmul(Ginv, c)
    elif len(b) == 1:
        lam = (-b - np.matmul(np.matmul(A, Ginv), c.T))/(np.matmul(np.matmul(A, Ginv),A.T))
        x = -np.matmul(Ginv, c.T) - lam*np.matmul(Ginv, A.T).T
    else:
        lam = np.linalg.solve((A@ Ginv)@A.T, -b - (A@Ginv)@ c)
        x = -np.matmul(Ginv, c) - np.matmul(np.matmul(Ginv, A.T), lam)
    return x



#regresa el valor de la menor alfa y su indice
def encuentraAlpha(Ad, bd, Wind, x, d):
    n = int(len(bd)- sum(Wind))
    alphas = np.zeros(n)
    k = 0
    aux1 = Ad@x
    aux2 = Ad@d
    for i in range(len(bd)):
        if Wind[i] == 0:
            if aux2[i] > 0:
                alphas[k] = (bd[i] - aux1[i])/aux2[i]
            else:
                alphas[k] = 2
            k = k+1
    j = np.argmin(alphas)
    s = -1
    i = -1
    resp = True
    while resp:
        i = i+1
        if Wind[i]==0:
            s = s+1
        if s == j:
            resp = False
    return min(alphas), i


# dados los multiplicadores de lagrange del conjunto W_k encuentra j y muj
def encuentraJ(multi, Wind, ni):
    if type(multi) == int:
        return 0,0
    else:
        aux = multi[ni:len(multi)]
        if aux.size == 0:
            mu = 0
            j = 0
        else:
            mu = np.min(aux)
            for j in range(len(multi)):
                if abs(aux[j]-mu) < 1e-8:
                    break
        k =0
        for i in range(len(Wind)):
            if Wind[i]==1:
                if k == j:
                    break
                k = k +1
        return mu, i

#Actualiza el conjunto W_k  para cuando lo piden las ramas 1 y 2
def actualizaW(Wind, Ai=0, Ad=0):
    W = np.copy(Ai)
    for i in range(len(Wind)):
        if(Wind[i]==1):
            if np.all(W == 0):
                W = Ad[i,:]
            else:
                W = np.vstack((W, Ad[i,:]))
    return W

#checa si W_k pertenece al conjunto activo y regresa Wind igual si W pretenece, modificada si Wind no pretenece
def checaW(x, Ad, bd, Wind, tol):
    n = len(bd)
    aux = np.matmul(Ad,x)
    for i in range(n):
        if Wind[i]==1:
            if aux[i] - bd[i] > tol:
                Wind[i] = 0
    return Wind


#Algoritmo del conjunto activo dado como la figura 2.1 de clase
def conjuntoActivo(G, c, x =0, const = 0, Ai=0, bi=0, Ad=0, bd=0, tol = 1e-9, iter = 100):
    rama1 = True
    rama2 = True
    if type(bi) == int:
        ni = 0
    else:
        ni = len(bi)
    if type(x)== int:
        x = encuentraFactible(Ai, bi, Ad, bd)
    print("x inicial = ", x)
    print("q(x) = ", .5*x.T@G@x + x.T@c + const, end = ' ')
    k = 0
    k2 = 0
    W, Wind = defW(x, Ai, bi, Ad, bd)
    gk = np.matmul(G,x) + c
    while rama2:
        if int(ni+ sum(Wind)) == 0:
            d = np.linalg.solve(G, -gk)#minimo si no se tienen restricciónes
        else:
            d = metodoRango(G, W, gk, np.zeros(int(ni+sum(Wind))))
        if max(abs(d)) < tol:
            rama1 = False
        while rama1:
            alpha, j = encuentraAlpha(Ad, bd, Wind, x, d)
            x = x + min(1, alpha)*d
            print("Rama 1", end = ' ')
            print("max d = ", max(abs(d)), end = ' ')
            print("q(x) = ", .5*x.T@G@x + x.T@c + const, end = ' ')
            print("alpha = ", alpha, end = ' ')
            print("x = ", x, end =' ')
            if alpha <= 1:
                print("j = ", j, end = ' ')
                Wind[j] = 1
                W = actualizaW(Wind, Ai, Ad)
            else:
                Wind = checaW(x, Ad, bd, Wind, tol)
                W = actualizaW(Wind, Ai, Ad)
            k = k + 1
            if k == iter:
                break
            
            gk = np.matmul(G,x) + c
            if int(ni+ sum(Wind)) == 0:
                d = np.linalg.solve(G, -gk)#minimo si no se tienen restricciónes
            else:
                d = metodoRango(G, W, gk, np.zeros(int(ni+sum(Wind))))
            if max(abs(d)) < tol:
                rama1 = False
        if k == iter or k2 == iter:
            break
        rama1 = True
        if int(ni+ sum(Wind)) == 1:
            aux = np.array([W[0]])
            for i in range(1,len(W)):
                aux = np.vstack((aux, W[i]))
            W = aux
            multi = scipy.linalg.lstsq(W, -gk)[0]
        elif int(ni+ sum(Wind)) == 0:
            multi = 0
        else:
            multi = scipy.linalg.lstsq(W.T, -gk)[0]#utiliza scipy.linalg.lstsq para encontras los multiplicadores de W_k los otros son cero por complementariedad
        mu, j = encuentraJ(multi, Wind, ni)
        if mu < 0:
            Wind[j] = 0
            W = actualizaW(Wind, Ai, Ad)
        else:
            rama2 = False
        if rama2:
            print("rama 2", end = ' ')
            print("j =", j, end = ' ')
            print("mu = ", mu)

            k2 = k2+1
    if int(ni+ sum(Wind)) == 1:
        aux = np.array([W[0]])
        for i in range(1,len(W)):
            aux = np.vstack((aux, W[i]))
        W = aux
        multi = scipy.linalg.lstsq(W, -gk)[0]
    elif int(ni+ sum(Wind)) == 0:
        multi = 0
    else:
        multi = scipy.linalg.lstsq(W.T, -gk)[0]
    print("numero de iteraciones =", k)
    print("q(x) = ", .5*x.T@G@x + x.T@c + const)
    return  x, multi, Wind

#Acomoda todos los multiplicadores de lagrange utilizando unicamente los de W_k
def encuentraMulti(multi, Wind, ni=0):
    aux = np.zeros(ni + len(Wind))
    k = 0
    for i in range(ni):
        aux[i] = multi[i]
    for i in range(len(Wind)):
        if Wind[i]==1:
            aux[i+ni]= multi[ni+k]
            k = k+1
    return aux




print("Ejercicio 2.6:\n")

G = 2*np.eye(2)
c = np.array([-2,-5])
Ad = np.array([[-1,2], [1,2], [1,-2], [-1,0], [0,-1]])
bd = np.array([2,6,2,0,0])
print("\n\n\n")
x, multi, Wind = conjuntoActivo(G,c, const = 7.25, Ad=Ad, bd = bd)
multi = encuentraMulti(multi, Wind)
print("x =", x, "multiplicadores =", multi)



# ejercicio en clase
print("Ejercicio en clase:\n")
G = np.eye(3)
c = -np.ones(3)
const = 1.5
Ai = np.array([1,1, 1])
bi = np.array([3])
Ad = -np.eye(3)
bd = np.zeros(3)
x = np.array([3,0,0])
x, multi, Wind = conjuntoActivo(G,c, x = x, const = 1.5, Ad=Ad, bd = bd, Ai = Ai, bi = bi)
multi = encuentraMulti(multi, Wind)
print("x =", x, "multiplicadores =", multi, "\n\n")



#klee minty

n =15
G = np.eye(n)
c = -np.ones(n)
Ad = np.eye(n)
A2 = -np.eye(n)
bd = np.zeros(n)
b2 = np.zeros(n)
for i in range(n):
    bd[i] = 2**(i+1)-1
    for j in range(n):
        if i-j>0:
            Ad[i,j]=2
bd[0]=1
Ad = np.vstack((Ad, A2))
bd = np.hstack((bd, b2))

"""
x, multi, Wind = conjuntoActivo(G,c, const = 7.25, Ad=Ad, bd = bd)
multi = encuentraMulti(multi, Wind)
print("x =", x)

"""
