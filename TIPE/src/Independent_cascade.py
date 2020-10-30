#import matplotlib.pyplot as plt
#import networkx as nx # Creation et affichage de graphes 
import scipy as sp # Utilisation des matrices 
import random
import time
import copy
from scipy.io import mmread
import multiprocessing
## Fonctions utilitaires

# Associe une proba a chaque arete
def probas_aleatoires(L):
    P = copy.deepcopy(L)
    n = len(L)
    for i in range(n):
        for j in range(len(L[i])):
            P[i][j] = (L[i][j],random.random())
    return P
    
def liste_adjacence(M, n):
    L = []
    for i in range(n):
        l = []
        for j in range(n):
            if M[i,j] > 0:
                l.append(j)
        L.append(l)
    return L
    
def degre(L, i):
    return len(L[i])
    
# Dessine le graphe
def dessiner(G, actifs, n):
    pos = nx.spring_layout(G)
    nx.draw(G, pos, node_color = 'b', node_size = 10)
    nx.draw_networkx_nodes(G, pos, nodelist = actifs, node_color = 'r', node_size = 20)

def charger_graphe(chemin, n):
    L = [[] for i in range(n)]
    f = open(chemin)
    ligne = f.readline().strip()
    while ligne != '':
        i1 = ligne.find(' ')
        i2 = ligne[i1+1:].find(' ') + i1+1
        u = int(ligne[:i1]) 
        v = int(ligne[i1+1:i2]) 
        L[u].append(v)
        ligne = f.readline().strip()
    return L
    
##Operations sur les tas

#Renvoie l'indice du père du noeud k
def pere(k):
    return (k-1)//2
    
def fils_gauche(k):
    return 2*k + 1

def fils_droit(k):
    return 2*k + 2

#Renvoie l'indice du plus grand élément entre les deux fils et le père
def max_pere_fils(tas, k):
    max = k
    n = len(tas)
    if n > fils_gauche(k) and tas[fils_gauche(k)] > tas[k]:
        max = fils_gauche(k)
    if n > fils_droit(k) and tas[fils_droit(k)] > tas[max]:
        max = fils_droit(k)
    return max

#On place l'élément à ajouter à la dernière place du tas puis on le fait remonter tant qu'il est plus grand que son père
def inserer(tas, indices, e):
    tas.append(e)
    v = e[1] 
    i = len(tas) - 1 #On place e en bas du tas
    indices[v] = i
    while(e > tas[pere(i)] and i > 0): #Tant que l'élément est plus grand que son père
        indices[v] = pere(i)
        indices[tas[pere(i)][1]] = i #On échange leurs indices
        tas[i], tas[pere(i)] = tas[pere(i)], tas[i] #Puis on les échange
        i = pere(i)
        
#On place le dernier élément du tas à la racine puis on le fait descendre tant qu'il est plus petit que le plus grand de ses fils
def retirer_max(tas, indices):
    racine = tas[0]
    dernier = tas[-1]
    tas[0] = dernier
    tas.pop()
    k = 0
    u = dernier[1]
    indices[u] = 0
    n = len(tas)
    while(tas[max_pere_fils(tas, k)] > tas[k] and k < n): #Tant que u est plus petit que le plus grand de ses fils
        max = max_pere_fils(tas, k)
        indices[tas[max][1]], indices[u] = k, max #On echange leurs indices
        tas[max], tas[k] = tas[k], tas[max] #Puis on les échange
        k = max
    return racine
    
def diminuer_clef(tas, indices, v, x):
    taille_tas = len(tas)
    i = indices[v]
    tas[i] = (tas[i][0] - x, v)
    while(tas[max_pere_fils(tas, i)] > tas[i] and i < taille_tas):
        max = max_pere_fils(tas, i)
        indices[v], indices[tas[max][1]] = max, i #On echange leurs indices
        tas[max], tas[i] = tas[i], tas[max] #Puis on les echange
        i = max
    
## Fonctions utiles a la simulation



def diffusion(L, p, A): #p : probabilité de propagation
    n = len(L)
    actifs = copy.deepcopy(A)
    nouveaux_actifs = copy.deepcopy(A)
    est_actif = [False for i in range(n)]
    for u in actifs:
        est_actif[u] = True
    while len(nouveaux_actifs) > 0:
        prochains_actifs = []
        for u in nouveaux_actifs:
            for v in L[u]:
                if not est_actif[v]: 
                    if random.random() <= p:
                        est_actif[v] = True
                        actifs.append(v)
                        prochains_actifs.append(v)
        nouveaux_actifs = copy.deepcopy(prochains_actifs)
    return actifs
    
    
    
    
# Calcule l'influence d'un ensemble de noeuds

def sigma(L, p, A, iterations): 
    s = 0
    for i in range(iterations):
        influence = len(diffusion(L, p, A))
        s += influence
    return s/iterations
    
def sigma_multiprocessing(args):
    L, p, A, iterations, v = args[0], args[1], args[2], args[3], args[4]
    return sigma(L, p, A, iterations), v
    
def sigma_tuple(args):
    return sigma(*args)
    
def diffusion_tuple(args):
    return diffusion(*args)

def sigma_parallele(L, p, A, iterations, pool):
    t0 = time.time()
    taches = [(L, p, A) for i in range(iterations)]
    resultats = pool.map_async(diffusion_tuple, taches, chunksize = iterations // 8).get()
    print(time.time() - t0)
    return sum(len(l) for l in resultats) / iterations
    
##Algorithmes
# Trouve le noeud qui maximise le gain marginal



def noeud_optimal(L, p, A, iterations): 
    noeud = 0
    max = 0
    n = len(L)
    for i in range(n):
        if i not in A : 
            s = sigma(L, p, A + [i], iterations)
            if s > max: 
                max = s
                noeud = i
    return max, noeud



def noeud_optimal_multiprocessing(L, p, A, iterations, pool):
    n = len(L)
    taches = []
    for v in range(n):
        if v not in A:
            taches.append((L, p, A + [v], iterations, v))
    influences = pool.map_async(sigma_multiprocessing, taches, chunksize = iterations // 8).get()
    return max(influences) #On renvoie le noeud associe a la plus grande influence

# Renvoie l'ensemble de noeuds qui maximise l'influence



 
def algo_glouton(L, p, k, iterations):
    A = [] 
    for i in range(k):
        x, influence = noeud_optimal(L, p, A, iterations)
        A.append(x)
    return A
    
    
    
    
def greedy_algo_parallele(L, p, k, iterations):
    A = [] 
    pool = multiprocessing.Pool(processes = 8)
    influences = []
    for i in range(k):
        t0 = time.time()
        print("nombre de noeuds trouves : ", i)
        influence, x = noeud_optimal_multiprocessing(L, p, A, iterations, pool)
        influences.append(influence)
        A.append(x)
        print(time.time() - t0)
    pool.close()
    pool.join()
    return A, influences
    
#Renvoie les k noeuds ayant les plus grands degres



def degre_max(L, k):
    n = len(L)
    tas = []
    indices = [0 for i in range(n)]
    choisis = []
    for v in range(n):
        inserer(tas, indices, (degre(L, v), v))
    for i in range(k):
        _, u = retirer_max(tas, indices)
        choisis.append(u)
    return choisis
    
    
    
def degree_discount(L, k, p):
    n = len(L)
    tas = []
    choisis = []
    degres = []
    indices = [0 for i in range(n)]
    nb_voisins_actifs = [0 for i in range(n)]
    for u in range(n):
        du = degre(L, u)
        inserer(tas, indices, (du, u))
        degres.append(du)
    for i in range(k):
        _, u = retirer_max(tas, indices)
        choisis.append(u)
        for v in L[u]:
            if v not in choisis: 
                nb_voisins_actifs[v] += 1
                dv = degres[v]
                tv = nb_voisins_actifs[v]
                nouveau_degre = dv - 2 * tv - (dv - tv) * tv * p 
                diminuer_clef(tas, indices, v, nouveau_degre)
        
    return choisis
    



"""
celf presentation:

def celf(L, k, p, iterations):
    n = len(L)
    gains_marginaux = []
    indices = [0 for i in range(n)]
    for v in range(n):
        s = (sigma(L, p, [v], iterations), v)
        inserer(gains_marginaux, indices, s)
    influence, noeud_max = retirer_max(gains_marginaux, indices)
    choisis = [noeud_max]
    for i in range(k-1):
        _, noeud_courant = retirer_max(gains_marginaux, indices)
        nouvelle_influence = sigma(L, p, choisis + [noeud_courant], iterations)
        inserer(gains_marginaux, indices, (nouvelle_influence - influence, noeud_courant))
        while gains_marginaux[0][1] != noeud_courant: #Tant que la racine ne reste pas racine
            _, noeud_courant = retirer_max(gains_marginaux, indices)
            nouvelle_influence = sigma(L, p, choisis + [noeud_courant], iterations)
            inserer(gains_marginaux, indices, (nouvelle_influence - influence, noeud_courant))
        influence = nouvelle_influence
        choisis.append(noeud_courant)
        retirer_max(gains_marginaux, indices)
    return choisis
"""

    
def celf(L, k, p, iterations):
    gains_marginaux = []
    n = len(L)
    indices = [0 for i in range(n)]
    pool = multiprocessing.Pool(processes = 8)
    marginaux_1 = []
    for v in range(n):
        print(v)
        s = (sigma_parallele(L, p, [v], iterations, pool), v)
        inserer(gains_marginaux, indices, s)
        marginaux_1.append(str(s[0]))
    influence, noeud_max = retirer_max(gains_marginaux, indices)
    choisis = [noeud_max]
    influences = [influence]
    lookups = []
    for i in range(k-1):
        print("recherche du noeud ", i)
        lookup = 0
        _, noeud_courant = retirer_max(gains_marginaux, indices)
        nouvelle_influence = sigma_parallele(L, p, choisis + [noeud_courant], iterations, pool)
        inserer(gains_marginaux, indices, (nouvelle_influence - influence, noeud_courant))
        while gains_marginaux[0][1] != noeud_courant: #Tant que la racine ne reste pas racine
            lookup += 1
            _, noeud_courant = retirer_max(gains_marginaux, indices)
            nouvelle_influence = sigma_parallele(L, p, choisis + [noeud_courant], iterations, pool)
            inserer(gains_marginaux, indices, (nouvelle_influence - influence, noeud_courant))
        influence = nouvelle_influence
        choisis.append(noeud_courant)
        retirer_max(gains_marginaux, indices)
        influences.append(influence)
        lookups.append(lookup)
    return influences
    
##
n = 1518
L = liste_adjacence(mmread("TIPE/Python/socfb-Simmons.mtx").toarray(), n)
p = 0.015
iterations = 1000
influences = []
for k in range(1,401):
    print(k)
    influences.append(sigma(L, p, degree_discount(L, k, p), iterations))
print(influences)
"""f = open("TIPE/Python/MIT.txt")
nombre = ""
for ligne in f:
    for ch in ligne:
        if ch == "," or ch == "]":
            marginaux.append(int(nombre))
            nombre = ""
        elif ch != "[" :
            nombre = nombre + ch"""

"""with open('simmons_influences.txt', 'w+') as f:
    for i in influences:
        f.write("%s\n" % i)
with open('simmons_maginaux.txt', 'w+') as f:
    for g in marginaux:
        f.write("%s\n" % g)"""




"""
iterations = [50, 100, 200, 300, 400, 500] + [1000*k for k in range(1, 31)]
pool = multiprocessing.Pool(processes = None)
A = degre_max(L, 100)
influences = []
for i in iterations:
    print(i)
    influences.append(sigma_parallelle(L, p, A, i, pool))
print(influences)

valeurs_k = [1] + [2*i for i in range(1, 24)]
temps = []
ensembles = []
for k in valeurs_k:
    t0 = time.time()
    ensembles.append((degree_discount(L, k, p), k))
    temps.append(time.time() - t0)
print("ensembles ok")
pool = multiprocessing.Pool(processes = None)
parametres = [(L, p, A, iterations, k) for A , k in ensembles]
influences = pool.map_async(sigma_multiprocessing, parametres, chunksize = 1).get() #Calcul des influences en parallele
print("influences : ", influences)
print("temps : ", temps)
pool.close()
pool.join()
"""





