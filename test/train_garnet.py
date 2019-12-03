import uproot
import networkx as nx
import math
from collections import Counter
import numpy as np
import sys, os
import scipy
import numba
from numba import types
from numba.typed import Dict
import matplotlib.pyplot as plt
import argparse

print("Importing Tensorflow (with built-in Keras)")
from tensorflow import keras
import tensorflow as tf
print("TF version is %s " %tf.__version__)
import pandas
import sklearn
import sklearn.metrics
import sklearn.ensemble
import scipy.sparse
import json
from keras_models import *

@numba.njit
def fill_target_matrix(mat, blids):
    for i in range(len(blids)):
        for j in range(i+1, len(blids)):
            if blids[i] == blids[j]:
                mat[i,j] = 1

@numba.njit
def fill_elem_pairs(elem_pairs_X, elem_pairs_y, elems, dm, target_matrix, skip_dm_0):
    n = 0
    for i in range(len(elems)):
        for j in range(i+1, len(elems)):
            if n >= elem_pairs_X.shape[0]:
                break
            if dm[i,j] > 0 or skip_dm_0==False:
                elem_pairs_X[n, 0] = elems[i, 0]
                elem_pairs_X[n, 1] = elems[i, 1]
                elem_pairs_X[n, 2] = elems[j, 0]
                elem_pairs_X[n, 3] = elems[j, 1]
                elem_pairs_X[n, 4] = dm[i,j]
                elem_pairs_y[n, 0] = int(target_matrix[i,j])
                n += 1
    return n

                
def load_file(fn):
    fi = uproot.open(fn)
    tree = fi.get("pftree")
    data = tree.arrays(tree.keys())
    data = {str(k): v for k, v in data.items()}
    
    #get the list of element (iblock, ielem) to candidate associations
    linktree = fi.get("linktree_elemtocand")
    data_elemtocand = linktree.arrays(linktree.keys())
    data_elemtocand = {str(k): v for k, v in data_elemtocand.items()}
    
    linktree2 = fi.get("linktree")
    data_elemtoelem = linktree2.arrays(linktree2.keys())
    data_elemtoelem = {str(k): v for k, v in data_elemtoelem.items()}

    return data, data_elemtocand, data_elemtoelem

#Create a graph where the nodes are elements and candidates and edges are
#whether or not a candidate was produced from a given set of elements
def create_graph_elements_candidates(data, data_elemtocand, iev):

    pfgraph = nx.Graph()
    node_pos = {}

    #Add nodes for calorimeter clusters
    for i in range(len(data["clusters_iblock"][iev])):
        ibl = data["clusters_iblock"][iev][i]
        iel = data["clusters_ielem"][iev][i]
        this = ("E", ibl, iel)
        node_pos[this] = data["clusters_eta"][iev][i], data["clusters_phi"][iev][i]
        pfgraph.add_node(this, type=data["clusters_type"][iev][i])
    
    #Add nodes for tracks
    for i in range(len(data["tracks_iblock"][iev])):
        ibl = data["tracks_iblock"][iev][i]
        iel = data["tracks_ielem"][iev][i]
        this = ("E", ibl, iel)
        node_pos[this] = data["tracks_outer_eta"][iev][i], data["tracks_outer_phi"][iev][i]
        if node_pos[this][0] == 0 and node_pos[this][1] == 0:
            node_pos[this] = data["tracks_inner_eta"][iev][i], data["tracks_inner_phi"][iev][i]
        if node_pos[this][0] == 0 and node_pos[this][1] == 0:
            node_pos[this] = data["tracks_eta"][iev][i], data["tracks_phi"][iev][i]
        pfgraph.add_node(this, type=1)
    
    #Add nodes for candidates 
    for i in range(len(data["pfcands_iblock"][iev])):
        this = ("C", i)
        node_pos[this] = data["pfcands_eta"][iev][i], data["pfcands_phi"][iev][i]
        pfgraph.add_node(this, type=-1)
    
    #Add edges between elements and candidates based on PFAlgo 
    for i in range(len(data_elemtocand["linkdata_elemtocand_ielem"][iev])):
        ibl = data_elemtocand["linkdata_elemtocand_iblock"][iev][i]
        iel = data_elemtocand["linkdata_elemtocand_ielem"][iev][i]
        ic = data_elemtocand["linkdata_elemtocand_icand"][iev][i]
        u = ("E", ibl, iel)
        v = ("C", ic)
        if u in pfgraph.nodes and v in pfgraph.nodes:
            p0 = node_pos[u]
            p1 = node_pos[v]
            dist = math.sqrt((p0[0]-p1[0])**2 + (p0[1]-p1[1])**2)
            pfgraph.add_edge(u, v, weight=dist)
    
    return pfgraph

def analyze_graph_subgraph_elements(pfgraph):
    sub_graphs = list(nx.connected_component_subgraphs(pfgraph)) #every candidate linked to elements
    subgraph_element_types = []

    elem_to_newblock = {}
    cand_to_newblock = {}
    for inewblock, sg in enumerate(sub_graphs): #For each subgraph (a connection of elements to pf cand.), loop over all nodes and add to elem_to_newblock set if not already on the list, meaning elements can only be connected to one pf candidates
        element_types = []
        for node in sg.nodes:
            node_type = node[0]
            if node_type == "E": #for clusters
                ibl, iel = node[1], node[2]
                k = (ibl, iel)
                assert(not (k in elem_to_newblock)) 
                elem_to_newblock[k] = inewblock
                element_type = sg.nodes[node]["type"]
                element_types += [element_type]
            elif node_type == "C":
                icand = node[1]
                k = icand
                assert(not (k in cand_to_newblock))
                cand_to_newblock[k] = inewblock
                
        element_types = tuple(sorted(element_types))
        subgraph_element_types += [element_types] 
    return subgraph_element_types, elem_to_newblock, cand_to_newblock

def assign_cand(iblocks, ielems, elem_to_newblock, _i):
    icands = elem_to_newblock.get((iblocks[_i], ielems[_i]), -1)
    return icands

@numba.njit
def fill_dist_matrix(dist_matrix, elem_blk, elem_ielem, nelem, elem_to_elem):
    for i in range(nelem):
        bl = elem_blk[i]
        iel1 = elem_ielem[i]
        for j in range(i+1, nelem):
            if elem_blk[j] == bl:
                iel2 = elem_ielem[j]
                k = (bl, iel1, iel2)
                if k in elem_to_elem:
                    dist_matrix[i,j] = elem_to_elem[k]
    
def prepare_data(data, data_elemtocand, data_elemtoelem, elem_to_newblock, cand_to_newblock, iev):
   
    #clusters
    X1 = np.vstack([ #For each event, a vector of cluster properties
        data["clusters_type"][iev],
        data["clusters_energy"][iev],
        data["clusters_eta"][iev],
        data["clusters_phi"][iev]]
    ).T
    ys1 = np.array([assign_cand( #For each event, a vector of pf candidate ID
        data["clusters_iblock"][iev],
        data["clusters_ielem"][iev],
        elem_to_newblock, i)
        for i in range(len(data["clusters_phi"][iev]))
    ])
    #tracks
    X2 = np.vstack([
        1*np.ones_like(data["tracks_qoverp"][iev]),
        data["tracks_qoverp"][iev],
        data["tracks_eta"][iev],
        data["tracks_phi"][iev],
        data["tracks_inner_eta"][iev],
        data["tracks_inner_phi"][iev],
        data["tracks_outer_eta"][iev],
        data["tracks_outer_phi"][iev]]
    ).T
    ys2 = np.array([assign_cand(
        data["tracks_iblock"][iev],
        data["tracks_ielem"][iev],
        elem_to_newblock, i)
    for i in range(len(data["tracks_phi"][iev]))])
    
    
    print(' How many tracks do I have  : X2.shape[0] = ',  X2.shape[0])
    print(' How many clusters do I have: X1.shape[0] = ',  X1.shape[0])
    print(' How many tracks features do I have: X2.shape[1] = ',  X2.shape[1])
    print(' How many clusterfeatures do I have: X1.shape[1] = ',  X1.shape[1])
    #make the track array the same size as the clusters, concatenate
    X1p = np.pad(X1, ((0,0),(0, X2.shape[1] - X1.shape[1])), mode="constant")
    X = np.vstack([X1p, X2])
    y = np.concatenate([ys1, ys2])

    #Fill the distance matrix between all elements
    nelem = len(X)
    dist_matrix = np.zeros((nelem, nelem))
    bls = data_elemtoelem["linkdata_iblock"][iev]
    el1 = data_elemtoelem["linkdata_ielem"][iev]
    el2 = data_elemtoelem["linkdata_jelem"][iev]
    dist = data_elemtoelem["linkdata_distance"][iev]

    elem_to_elem = {(bl, e1, e2): d for bl, e1, e2, d in zip(bls, el1, el2, dist)}
    elem_to_elem_nd = Dict.empty(
        key_type=types.Tuple([types.uint32, types.uint32, types.uint32]),
        value_type=types.float64
    )
    for k, v in elem_to_elem.items():
        elem_to_elem_nd[k] = v

    elem_blk = np.hstack([data["clusters_iblock"][iev], data["tracks_iblock"][iev]])
    elem_ielem = np.hstack([data["clusters_ielem"][iev], data["tracks_ielem"][iev]])

    fill_dist_matrix(dist_matrix, elem_blk, elem_ielem, nelem, elem_to_elem_nd)
    dist_matrix_sparse = scipy.sparse.dok_matrix(dist_matrix)
        
    cand_data = np.vstack([
        data["pfcands_pdgid"][iev],
        data["pfcands_pt"][iev],
        data["pfcands_eta"][iev],
        data["pfcands_phi"][iev],
    ]).T
    cand_block_id = np.array([cand_to_newblock.get(ic, -1) for ic in range(len(data["pfcands_phi"][iev]))], dtype=np.int64)
    
    genpart_data = np.vstack([
        data["genparticles_pdgid"][iev],
        data["genparticles_pt"][iev],
        data["genparticles_eta"][iev],
        data["genparticles_phi"][iev],
    ]).T
    
    return X, y, cand_data, cand_block_id, dist_matrix_sparse

def get_unique_X_y(X, Xbl, y, ybl, maxn=3):
    uniqs = np.unique(Xbl)
    
    Xs = []
    ys = []
    for bl in uniqs:
        subX = X[Xbl==bl]
        suby = y[ybl==bl]
        
        #choose only miniblocks with up to 3 elements to simplify the problem
        if len(suby) > len(subX):
            print("Odd event with more candidates than elements in block", len(suby), len(subX))

        if len(subX) > maxn:
            continue
        if len(suby) > maxn:
            continue
        subX = np.pad(subX, ((0, maxn - subX.shape[0]), (0,0)), mode="constant")
        suby = np.pad(suby, ((0, maxn - suby.shape[0]), (0,0)), mode="constant")
        
        Xs += [subX]
        ys += [suby]
        
    return Xs, ys

def build_input_graph(infolder,outfolder):
  if not os.path.exists(outfolder):
      os.makedirs(outfolder)
          
  files = []
  path = infolder
  for r,d,f in os.walk(path):
    for file in f:
      if '.root' in file:
        print("Appending file to list: %s" %file)
        files.append(os.path.join(r, file))
  fn = files[0]
  data, data_elemtocand, data_elemtoelem = load_file(fn)

  nev = len(data["pfcands_pt"])
  print("Loaded {0} events".format(nev))

  all_sgs = []
  for iev in range(nev):
      print("{0}/{1}".format(iev, nev))

      #Create a graph of the elements and candidates
      pfgraph = create_graph_elements_candidates(data, data_elemtocand, iev)
      #Find disjoint subgraphs
      sgs, elem_to_newblock, cand_to_newblock = analyze_graph_subgraph_elements(pfgraph)
      
      #Create arrays from subgraphs
      elements, block_id, pfcands, cand_block_id, dist_matrix = prepare_data(
          data, data_elemtocand, data_elemtoelem, elem_to_newblock, cand_to_newblock, iev)

      #save the all the elements, candidates and the miniblock id
      cache_filename = 'data/'+fn.split("/")[8].replace(".root", "_ev{0}.npz".format(iev))
      print cache_filename
      with open(cache_filename, "wb") as fi:
          np.savez(fi, elements=elements, element_block_id=block_id, candidates=pfcands, candidate_block_id=cand_block_id)

      cache_filename = 'data/'+ fn.split("/")[8].replace(".root", "_dist{0}.npz".format(iev))
      with open(cache_filename, "wb") as fi:
          scipy.sparse.save_npz(fi, dist_matrix.tocoo())
  
      # #save the miniblocks separately (Xs - all miniblocks in event, ys - all candidates made from each block) 
      # Xs, ys = get_unique_X_y(elements, block_id, pfcands, cand_block_id)
      # cache_filename = fn.replace(".root", "_cl{0}.npz".format(iev))
      # with open(cache_filename, "wb") as fi:
      #     np.savez(fi, Xs=Xs, ys=ys)

      all_sgs += sgs
 
  block_sizes = Counter([len(sg) for sg in all_sgs])
  print("For all events, print how many elements is in each subgraph(block)")
  print("block sizes", block_sizes)

  for blocksize in range(1,5):
      print("For all events, how many subgraphs/blocks have %i elements" %blocksize)
      blocks_nelem = Counter([tuple(sg) for sg in all_sgs if len(sg)==blocksize])
      print("{0}-element blocks".format(blocksize), blocks_nelem)
      
#Given an event file, creates a list of all the elements pairs that have a non-infinite distance as per PFAlgo
#Will produce the X vector with [n_elem_pairs, 3], where the columns are (elem1_type, elem2_type, dist)
#and an y vector (classification target) with [n_elem_pairs, 1], where the value is 0 or 1, depending
#on if the elements are in the same miniblock according to PFAlgo
def load_element_pairs(fn):

    #Load the elements
    fi = open(fn, "rb")
    data = np.load(fi)
    els = data["elements"]
    els_blid = data["element_block_id"]

    #Load the distance matrix
    fi = open(fn.replace("ev", "dist"), "rb")
    dm = scipy.sparse.load_npz(fi).todense()
    
    #Create the matrix of elements that are connected according to the miniblock id
    target_matrix = np.zeros((len(els_blid), len(els_blid)), dtype=np.int32)
    fill_target_matrix(target_matrix, els_blid)

    #Fill the element pairs
    elem_pairs_X = np.zeros((20000,5), dtype=np.float32) #el 1, eta1, el2, eta2, dEta?
    elem_pairs_y = np.zeros((20000,1), dtype=np.float32) #truth
    n = fill_elem_pairs(elem_pairs_X, elem_pairs_y, els, dm, target_matrix, True)
    elem_pairs_X = elem_pairs_X[:n] #strip last unneccessary rows
    elem_pairs_y = elem_pairs_y[:n]
    return elem_pairs_X, elem_pairs_y

def getModel_garnet(inputShape):
  model = GarNetClusteringModel()
  model.build(input_shape=(1,inputShape.shape[0],inputShape.shape[1])) #BxVxF: batch, vertices(==elements per event), features (pt,eta,phi)
  print model
  opt = keras.optimizers.Adam(lr=1e-3)
  model.compile(loss="binary_crossentropy", optimizer=opt)
  model.summary()
  return 
def getModel_baseline(inputShape):
  
  nunit = 256
  dropout = 0.2
  model = keras.models.Sequential()
  model.add(keras.layers.Dense(nunit, input_shape=(inputShape.shape[1], )))
  
  model.add(keras.layers.LeakyReLU())
  model.add(keras.layers.Dropout(dropout))
  model.add(keras.layers.Dense(nunit))
  model.add(keras.layers.BatchNormalization())
  
  model.add(keras.layers.LeakyReLU())
  model.add(keras.layers.Dropout(dropout))
  model.add(keras.layers.Dense(nunit))
  model.add(keras.layers.BatchNormalization())
  
  model.add(keras.layers.LeakyReLU())
  model.add(keras.layers.Dropout(dropout))
  model.add(keras.layers.Dense(nunit))
  model.add(keras.layers.BatchNormalization())
  
  model.add(keras.layers.LeakyReLU())
  model.add(keras.layers.Dropout(dropout))
  model.add(keras.layers.Dense(nunit))
  model.add(keras.layers.BatchNormalization())
  
  model.add(keras.layers.LeakyReLU())
  model.add(keras.layers.Dropout(dropout))
  model.add(keras.layers.Dense(nunit))
  model.add(keras.layers.BatchNormalization())
  
  model.add(keras.layers.LeakyReLU())
  model.add(keras.layers.Dense(1, activation="sigmoid"))
  
  opt = keras.optimizers.Adam(lr=1e-3)
  
  model.compile(loss="binary_crossentropy", optimizer=opt)
  model.summary()
  
  return model
    
def train(type,infolder):
  
  try:
      import setGPU
  except:
      print("Could not import setGPU, Nvidia device not found")
  
  gpuFraction = 0.2
  gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=gpuFraction)
  sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
  tf.compat.v1.keras.backend.set_session(sess)
  print('using gpu memory fraction: '+str(gpuFraction))
  

  all_elem_pairs_X = []
  all_elem_pairs_y = []
  
  files = []
  path = infolder
  for r,d,f in os.walk(path):
    for file in f:
      if 'ev' in file:
        files.append(os.path.join(r, file))
        break

  for i,fn in enumerate(files):
    sys.stdout.write("\r" + "Loading file %i/%i"%(i,len(files)))
    sys.stdout.flush()
    elem_pairs_X, elem_pairs_y = load_element_pairs(fn)
    all_elem_pairs_X += [elem_pairs_X]
    all_elem_pairs_y += [elem_pairs_y]

  elem_pairs_X = np.vstack(all_elem_pairs_X)
  elem_pairs_y = np.vstack(all_elem_pairs_y)

  shuf = np.random.permutation(range(len(elem_pairs_X)))
  elem_pairs_X = elem_pairs_X[shuf]
  elem_pairs_y = elem_pairs_y[shuf]

  weights = np.zeros(len(elem_pairs_y))
  ns = np.sum(elem_pairs_y[:, 0]==1)
  nb = np.sum(elem_pairs_y[:, 0]==0)
  weights[elem_pairs_y[:, 0]==1] = 1.0/ns
  weights[elem_pairs_y[:, 0]==0] = 1.0/nb
  
  modelname = 'garnet'
  if type.find('baseline')!=-1:
    modelname = 'clustering'
    model = getModel_baseline(elem_pairs_X)
  else:
    model = getModel_garnet(elem_pairs_X)
  ntrain = int(0.8*len(elem_pairs_X))
  ret = model.fit(
      elem_pairs_X[:ntrain], elem_pairs_y[:ntrain, 0], sample_weight=weights[:ntrain],
      validation_data=(elem_pairs_X[ntrain:], elem_pairs_y[ntrain:, 0], weights[ntrain:]),
      batch_size=10000, epochs=100
  )

  pp = model.predict(elem_pairs_X, batch_size=10000)
  confusion = sklearn.metrics.confusion_matrix(elem_pairs_y[ntrain:, 0], pp[ntrain:]>0.5)
  print("For %s" %modelname)
  print("Print confusion matrix:")
  print("[ [ True Pos     False Pos]")
  print("  [ False Neg    True Neg ] ]")
  print(confusion)

  training_info = {
      "loss": ret.history["loss"],
      "val_loss": ret.history["val_loss"]
  }

  with open(modelname+".json", "w") as fi:
      json.dump(training_info, fi)
  model.save(modelname+".h5")

def getROCplot():
	plt.clf()
	plt.figure()
	plt.xlim([0.05, 1.0])
	plt.ylim([0.0001, 0.7])
	plt.yscale('log')
	plt.grid()
	return plt
      
def doROC(infolder):
  all_elem_pairs_X = []
  all_elem_pairs_y = []
  
  files = []
  path = infolder
  for r,d,f in os.walk(path):
    for file in f:
      if 'ev' in file:
        files.append(os.path.join(r, file))

  for fn in files:
    elem_pairs_X, elem_pairs_y = load_element_pairs(fn)
    all_elem_pairs_X += [elem_pairs_X]
    all_elem_pairs_y += [elem_pairs_y]
  
  elem_pairs_X = np.vstack(all_elem_pairs_X)
  elem_pairs_y = np.vstack(all_elem_pairs_y)
  
  shuf = np.random.permutation(range(len(elem_pairs_X)))
  elem_pairs_X = elem_pairs_X[shuf]
  elem_pairs_y = elem_pairs_y[shuf]
  ntrain = int(0.8*len(elem_pairs_X))
  model = getModel_baseline(elem_pairs_X.shape[1])
  weightfiles = ['/afs/cern.ch/user/t/thaarres/public/pfstudies/clustering.h5','/eos/user/j/jpata/particleflow/clustering.h5']  
  colors = ['#377eb8', '#ff7f00', '#4daf4a','#f781bf', '#a65628', '#984ea3','#999999', '#e41a1c', '#dede00']
  model_names = ["GarNet","Baseline"]
  
  line_styles = ["solid", ":", "-."] * 30
  rocplt = getROCplot()
  
  for model_name, weightfile in zip(model_names,weightfiles):
    print("Loading weights for model %s" %model_name)
    # Loads the weights
    model.load_weights(weightfile)

    # Re-evaluate the model  
    # pp = model.predict(elem_pairs_X, batch_size=10000)
#     confusion = sklearn.metrics.confusion_matrix(elem_pairs_y[ntrain:, 0], pp[ntrain:]>0.5)
#     print("Print confusion matrix:")
#     print("[ [ True Pos     False Pos]")
#     print("  [ False Neg    True Neg ] ]")
#     print(confusion)
#     print(model.metrics_names)
#     loss = model.evaluate(elem_pairs_X,elem_pairs_y, batch_size=10000)
#     print("Restored model, loss: {:5.2f}".format(loss ))
  
    from sklearn.metrics import roc_curve,roc_auc_score
  
    y_val_cat_prob=model.predict_proba(elem_pairs_X)
    fpr , tpr , thresholds = roc_curve ( elem_pairs_y , y_val_cat_prob)  
  
    AOC = roc_auc_score(elem_pairs_y , y_val_cat_prob)
    print("AOC: {0}".format(AOC))
  
    fpr[fpr < 0.0001] = 0.0001
    rocplt.plot(tpr, fpr, color=colors.pop(0), lw=1, label='{0} (area = {1:.2f})'.format(model_name, AOC))
  
  rocplt.legend(loc="lower right")
  rocplt.savefig("ROC.png")
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--doInputGraphs', action='store_true')
    parser.add_argument('-r', '--doROC', action='store_true')
    parser.add_argument('-m', dest='model', type=str, default='garnet')
    parser.add_argument('-o', dest='output', type=str, default='data/')
    parser.add_argument('-i', dest='input', type=str, default="/eos/user/j/jpata/particleflow/TTbar/191009_155100/")
    args = parser.parse_args()
    if args.doInputGraphs:
      print("Building input graphs and saving to %s" %args.output)
      build_input_graph(args.input,args.output)
    elif args.doROC:
      doROC(args.output)
    else:  
      print("Opening .npz files in %s and training network %s" %(args.output, args.model ))
      train(args.model,args.output)
