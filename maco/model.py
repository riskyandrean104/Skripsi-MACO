import pandas as pd
import numpy as np
import random
import googlemaps
import copy
from numpy import inf

API_KEY = 'AIzaSyDIC0OWokoAFBc_EAiojs9A1RmeFNUjBK4'#enter Google Maps API key
gmaps = googlemaps.Client(key=API_KEY)

def distancematrix(data):
  a=[]
  for i in range(len(data.iloc[:,:-1])):
    b=[]
    for j in range(len(data.iloc[:,:-1])):
      distance = gmaps.distance_matrix(data.iloc[i,:-1], data.iloc[j,:-1],mode='driving')["rows"][0]["elements"][0]["distance"]["value"]
      b.append(distance)
    a.append(b)
  jarak_titik = pd.DataFrame(np.array(a))
  return jarak_titik

class Fuzzy:
    def __init__(self, data):
        #number of data

        self.jarak_titik = distancematrix(data)
        self.n = len(self.jarak_titik)

        #number of clusters
        self.k = 4

        #dimension of cluster
        self.d = len(data)

        # m parameter
        self.m = 2

        #number of iterations FCM
        self.MAX_ITERS = 100

    def initializeMembershipWeights(self):
        weight = np.random.dirichlet(np.ones(self.k),self.n)
        weight_arr = np.array(weight)
        return weight_arr

    def computeCentroids(self,weight_arr):
        C = []
        for i in range(self.k):
            weight_sum = np.power(weight_arr[:,i],self.m).sum()
            Cj = []
            for x in range(self.d):
                numerator = (self.jarak_titik.iloc[:,x].values * np.power(weight_arr[:,i],self.m)).sum()
                c_val = numerator/weight_sum
                Cj.append(c_val)
            C.append(Cj)
        return C
    
    def updateWeights(self,weight_arr,C):
        denom = np.zeros(self.n)
        for i in range(self.k):
            dist = (self.jarak_titik.iloc[:,:].values - C[i])**2
            dist = np.sum(dist, axis=1)
            dist = np.sqrt(dist)
            denom  = denom + np.power(1/dist,1/(self.m-1))

        for i in range(self.k):
            dist = (self.jarak_titik.iloc[:,:].values - C[i])**2
            dist = np.sum(dist, axis=1)
            dist = np.sqrt(dist)
            weight_arr[:,i] = np.divide(np.power(1/dist,1/(self.m-1)),denom)
        return weight_arr

    def ValuesClusterFinal(self,weight_arr,C):
        clusters_final = []
        nilai_cluster=[]
        for i in range(self.n):
            Max_val, idx = max((C, idx) for (idx, C) in enumerate(weight_arr[i]))
            nilai_cluster.append(Max_val)
            clusters_final.append(idx)
        values_cluster_final = np.stack((nilai_cluster,clusters_final),axis=1)
        return values_cluster_final
    
    def FuzzyMeansAlgorithm(self):
        weight_arr = self.initializeMembershipWeights()
        for z in range(self.MAX_ITERS):
            C = self.computeCentroids(weight_arr)
            weight_final = self.updateWeights(weight_arr,C)
            cluster_final = self.ValuesClusterFinal(weight_final,C)
        return C, weight_final, cluster_final


class ACO:
    def __init__(self, data):
        self.jarak_titik = distancematrix(data)
        self.iteration = 100 #ACO
        self.n_ants = len(data) #ACO
        self.n_citys = len(data) #ACO

        self.alpha = 5 #pheromone factor
        self.beta = 0.3 #visibility factor
        self.e = 0.9 #evaporation rate

        #menghitung visibilitas dari kota selanjutnya visibility(i,j)=1/d(i,j)

        self.visibility = 1/self.jarak_titik
        self.visibility[self.visibility == inf ] = 0

        #intializing pheromne present at the paths to the cities
        self.pheromne = .1*np.ones((self.n_ants,self.n_citys))

        #inisialisasi rute dari semut dengan ukuran (n_ants,n_citys+1) 
        #menambahkan 1 kota dikarenakan tsp akan kembali ke titik awal
        self.rute = np.ones((self.n_ants,self.n_citys+1))

        _, _, self.cluster_final = Fuzzy(data).FuzzyMeansAlgorithm()

    def MACO(self, data):
        for _ in range(self.iteration):
            
            self.rute[:,0] = 1          #inisialisasi titik awal dan akhir dari setiap semut '1' city '1'
            
            for i in range(self.n_ants):
                
                temp_visibility = np.array(self.visibility)         #menyimpan nilai visibility pada temporary visibility
                
                for j in range(self.n_citys-1):
                    
                    combine_feature = np.zeros(len(data))     #inisialisasi nilai combine feature berupa array dengan nilai 0 sebanyak jumlah data
                    cum_prob = np.zeros(len(data))            #inisialisasi nilai cumulative probability berupa array dengan nilai 0 sebanyak jumlah data
                    
                    cur_loc = int(self.rute[i,j]-1)        #lokasi semut sekarang
                    
                    temp_visibility[:,cur_loc] = 0     #membuat visibility dari lokasi semut menjadi 0
                    
                    p_feature = np.power(self.pheromne[cur_loc,:],self.beta)         #menghitung pheromone fitur 
                    v_feature = np.power(temp_visibility[cur_loc,:],self.alpha)  #menghitung visibility fitur
                    
                    p_feature = p_feature[:,np.newaxis]                     
                    v_feature = v_feature[:,np.newaxis]                     
                    
                    combine_feature = np.multiply(p_feature,v_feature)     #calculating the combine feature
                                
                    total = np.sum(combine_feature)                        #sum of all the feature
                    
                    probs = combine_feature/total   #finding probability of element probs(i) = comine_feature(i)/total
                    cum_prob = np.cumsum(probs)     #calculating cummulative sum

                    random_num = self.cluster_final[i][0]
                    r = np.random.random_sample()

                    city = np.nonzero(cum_prob>random_num)[0][0]+1   #mencari kota berikutnya dengan probabilitas tertinggi random(r)
                    
                    self.rute[i,j+1] = city              #menambah kota ke rute
                    
            
                left = list(set([i for i in range(1,self.n_citys+1)])-set(self.rute[i,:-2]))[0]     #finding the last untraversed city to route
                self.rute[i,-2] = left                   #adding untraversed city to route

            rute_opt = np.array(self.rute)               #inisialisasi optimal route
            
            dist_cost = np.zeros((self.n_ants,1))             
            
            for i in range(self.n_ants):
            
                s = 0
                for j in range(self.n_citys-1):
                    
                    s = s + self.jarak_titik.loc[int(rute_opt[i,j])-1,int(rute_opt[i,j+1])-1]   #menghitung panjang tour dari setiap semut

                dist_cost[i]=s                      #mengurutkan panjang tour/rute yang dihasilkan oleh setiap semut
                
            dist_min_loc = np.argmin(dist_cost)             #menemukan lokasi rute terpendek
            dist_min_cost = dist_cost[dist_min_loc]         #finging min of dist_cost
            best_route = self.rute[dist_min_loc,:]               #intializing current traversed as best route
            self.pheromne = (1-self.e)*self.pheromne                       #evaporation of pheromne with (1-e)

            for i in range(self.n_ants):
                for j in range(self.n_citys-1):
                    dt = 1/dist_cost[i]
                    self.pheromne[int(rute_opt[i,j])-1,int(rute_opt[i,j+1])-1] = self.pheromne[int(rute_opt[i,j])-1,int(rute_opt[i,j+1])-1] + dt   
                    #updating tpheromne dengan delta_distance'''
        Cost = (int(dist_min_cost[0]) + self.jarak_titik.loc[int(best_route[-2])-1,0])/1000
        return rute_opt, best_route, Cost

def rute(data):
    _, best_route, _ = ACO(data).MACO(data)
    rute_coordinate = []
    rute_location = []
    for i in best_route:
        i = i-1
        wp = data['Coordinate'][int(i)]
        loc = data['Lokasi'][int(i)]
        rute_coordinate.append(wp)
        rute_location.append(loc)
    rute = pd.DataFrame(np.stack((rute_coordinate,rute_location),axis=1), columns=["Coordinate", "Lokasi"])
    return rute

def directions(data):
    travel_time = 0
    distance_route = 0
    urutan_rute = rute(data)

    results = []
    
    for i in range(len(urutan_rute)):
        if i >= len(urutan_rute)-1:
            pass
        else:
            add_result = gmaps.directions(origin = urutan_rute['Coordinate'][i],destination = urutan_rute['Coordinate'][i+1])
            results.append(add_result)
            # for i, leg in enumerate(add_result[0]["legs"]):
            #     print(leg["start_address"], 
            #         "==>",
            #         leg["end_address"], 
            #         "distance: ",  
            #         leg["distance"]["value"], 
            #         "traveling Time: ",
            #         leg["duration"]["value"])
            #     for step in leg['steps']:
            #         html_instructions = step['html_instructions']
            #         print(html_instructions)

                    #=============

            # print()
            # travel_time += leg["duration"]["value"]
            # distance_route += leg["distance"]["value"]
    return results