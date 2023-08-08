#import matplotlib.pyplot as plt
#import networkx as nx # Creation et affichage de graphes 
import scipy as sp # Utilisation des matrices 
import random
import time
import copy
from scipy.io import mmread
import multiprocessing
## Fonctions utilitaires

# Cree la matrice des poids entre les aretes
def poids_aleatoires(L, n):
    for i in range(n):
        for j in range(len(L[i])):
            L[i][j] = (L[i][j],1/max(degre(L, i), degre(L, j)))
            
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
    pos = nx.random_layout(G)
    nx.draw(G, pos, node_color = 'b')
    nx.draw_networkx_nodes(G, pos, nodelist = actifs, node_color = 'r')
    

## Fonctions utiles a la simulation



def diffusion(L, A): 
    """
    L : liste d'adjacence contenant les couples (voisins,poids)
    A : ensemble de depart 
    renvoie la liste des noeuds actifs à la fin du processus
    """
    n = len(L)
    seuils = sp.random.rand(n)
    somme = [0 for i in range(n)]
    actifs = copy.deepcopy(A)
    est_actif = [False for i in range(n)]
    for u in actifs: 
        est_actif[u] = True
    nouveaux_actifs = copy.deepcopy(A)
    #Tant que des noeuds deviennent actifs on continue: 
    while len(nouveaux_actifs) > 0: 
        prochains_actifs = []
        """Pour chaque noeud qui vient d'être activé
        on essaye d'activer ses voisins"""
        for u in nouveaux_actifs:
            for v, p in L[u]:
                if not est_actif[v]:
                    somme[v] += p
                    if somme[v] >= seuils[v]:
                        est_actif[v] = True
                        actifs.append(v)
                        prochains_actifs.append(v)
        nouveaux_actifs = copy.deepcopy(prochains_actifs)
    return actifs





# Calcule l'influence d'un ensemble de noeuds
def sigma(L, A, iterations): 
    s = 0
    for i in range(iterations):
        influence = len(diffusion(L, A))
        s += influence
    return s/iterations
    
##Operations sur les tas
#Renvoie l'indice du pere du noeud k
def pere(k):
    return (k-1)//2
    
def fils_gauche(k):
    return 2*k + 1

def fils_droit(k):
    return 2*k + 2

#Renvoie l'indice du plus grand element entre les deux fils et le pere
def max_pere_fils(tas, k):
    max = k
    n = len(tas)
    if n > fils_gauche(k) and tas[fils_gauche(k)] > tas[k]:
        max = fils_gauche(k)
    if n > fils_droit(k) and tas[fils_droit(k)] > tas[max]:
        max = fils_droit(k)
    return max

#On place l'element a ajouter a la derniere place du tas puis on le fait remonter tant qu'il est plus grand que son pere
def inserer(tas, indices, e):
    tas.append(e)
    v = e[1] 
    i = len(tas) - 1 #On place e en bas du tas
    indices[v] = i
    while(e > tas[pere(i)] and i > 0): #Tant que l'element est plus grand que son pere
        indices[v] = pere(i)
        indices[tas[pere(i)][1]] = i #On echange leurs indices
        tas[i], tas[pere(i)] = tas[pere(i)], tas[i] #Puis on les echange
        i = pere(i)
        
#On place le dernier element du tas a la racine puis on le fait descendre tant qu'il est plus petit que le plus grand de ses fils
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
        tas[max], tas[k] = tas[k], tas[max] #Puis on les echange
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
    
##Algorithmes
# Trouve le noeud qui maximise le gain marginal
def noeud_optimal(L, A, iterations): 
    noeud = 0
    max = 0
    n = len(L)
    for i in range(n):
        if i not in A : # On ne considere que les noeuds qui n'ont pas encore ete choisis
            s = sigma(L, A + [i], iterations)
            if s > max: 
                max = s
                noeud = i
    return noeud, max

# Renvoie l'ensemble de noeuds qui maximise l'influence 
def greedy_algo(L, k, iterations):
    A = [] 
    for i in range(k):
        x, influence = noeud_optimal(L, A, iterations)
        A.append(x)
    return A, influence
#Renvoie les k noeuds ayant les plus grands degres
def degre_max(L, k):
    tas = []
    choisis = []
    for i in range(n):
        inserer(tas, (degre(L, i), i))
    for i in range(k):
        _, u = retirer_max(tas)
        choisis.append(u)
    return choisis

def single_discount(L, k):
    n = len(L)
    tas = []
    choisis = []
    indices = [0 for i in range(n)]
    degres = []
    for u in range(n):
        du = degre(L, u)
        inserer(tas, indices, (du, u))
        degres.append(du)
    for i in range(k):
        _, u = retirer_max(tas, indices)
        for v, p in L[u]:
            diminuer_clef(tas, indices, v, degres[v] - 1)
            degres[v] = degres[v] - 1
        choisis.append(u)
    return choisis

def sigma_multiprocessing(args):
    L, A, iterations, v = args[0], args[1], args[2], args[3]
    return sigma(L, A, iterations), v
    
def noeud_optimal_multiprocessing(L, A, iterations, pool):
    n = len(L)
    taches = []
    for v in range(n):
        if v not in A:
            taches.append((L, A + [v], iterations, v))
    influences = pool.map_async(sigma_multiprocessing, taches, chunksize = iterations // multiprocessing.cpu_count()).get()
    return max(influences) #On renvoie la plus grande influence ainsi que le noeud associe
    
def greedy_algo_parallele(L, k, iterations):
    A = [] 
    pool = multiprocessing.Pool(processes = 2)
    for i in range(k):
        t0 = time.time()
        print("nombre de noeuds trouves : ", i)
        influence, x = noeud_optimal_multiprocessing(L, A, iterations, pool)
        A.append(x)
        print(time.time() - t0)
    pool.close()
    pool.join()
    return A, influence
    
def diffusion_tuple(args):
    return diffusion(*args)

def sigma_parallele(L, A, iterations, pool):
    t0 = time.time()
    taches = [(L, A) for i in range(iterations)]
    resultats = pool.map_async(diffusion_tuple, taches, chunksize = iterations // 4).get()
    print(time.time() - t0)
    return sum(len(l) for l in resultats) / iterations
    
def celf(L, k, iterations):
    gains_marginaux = []
    n = len(L)
    indices = [0 for i in range(n)]
    marginaux_1 = []
    pool = multiprocessing.Pool(processes = 4)
    for v in range(n):
        print(v)
        s = (sigma_parallele(L, [v], iterations, pool), v)
        inserer(gains_marginaux, indices, s)
        marginaux_1.append(str(s[0]))
    influence, noeud_max = retirer_max(gains_marginaux, indices)
    choisis = [noeud_max]
    influences = [influence]
    for i in range(k-1):
        print("recherche du noeud ", i)
        _, noeud_courant = retirer_max(gains_marginaux, indices)
        nouvelle_influence = sigma_parallele(L, choisis + [noeud_courant], iterations, pool)
        inserer(gains_marginaux, indices, (nouvelle_influence - influence, noeud_courant))
        while gains_marginaux[0][1] != noeud_courant: #Tant que la racine ne reste pas racine
            _, noeud_courant = retirer_max(gains_marginaux, indices)
            nouvelle_influence = sigma_parallele(L, choisis + [noeud_courant], iterations, pool)
            inserer(gains_marginaux, indices, (nouvelle_influence - influence, noeud_courant))
        influence = nouvelle_influence
        choisis.append(noeud_courant)
        retirer_max(gains_marginaux, indices)
        influences.append(influence)
    return influences, gains_marginaux, marginaux_1




##
n = 1518
L = liste_adjacence(mmread("TIPE/Python/socfb-Simmons.mtx").toarray(), n)
poids_aleatoires(L, n)
iterations = 1000
t0 = time.time()
influences, gains_marginaux, marginaux_1 = celf(L, 200, iterations)
print(influences)






    

    