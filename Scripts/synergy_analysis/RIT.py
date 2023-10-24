'''
Random Insection Trees
http://jmlr.org/papers/volume15/shah14a/shah14a.pdf
'''
import numpy as np
import pandas as pd
from collections import defaultdict
import xlrd
import xlwt
import random

# random sample
class Node():
  '''
  A node in an intersection tree
  '''
  def __init__(self, parent):
    if (parent is None):
      self.active_array = None
      self.depth = 0
    else:
      self.active_array = parent.active_array
      self.depth = parent.depth + 1

  def activate(self, data):
    if (self.active_array is None):
      self.active_array = data
      return self.active_array
    else:
      insected_array = []
      for i, j in zip(self.active_array, data):
        if (i == j):
          if (i == 1):
            insected_array.append(1)
          else:
            insected_array.append(0)
        else:
          insected_array.append(0)
        self.active_array = np.array(insected_array)
        #self.active_array = set(self.active_array).intersection(set(data))
      return self.active_array
  def contrast(self, data):
      if (self.active_array is None):
        self.active_array = data
        return self.active_array
      else:
        diff_array = []
        for i, j in zip(self.active_array, data):
          if (i != j):
            diff_array.append(1)
          else:
            diff_array.append(0)
        self.active_array = np.array(diff_array)
        #self.active_array = set(self.active_array) - set(data)
        return self.active_array
        
  def get_depth(self):
    return self.depth

  def stop(self,positive_samples_array, negative_samples_array):
    length1 = len(positive_samples_array)
    length2 = len(negative_samples_array)
    #active_set = set(self.active_array
    #fre1 = sum([1 for active_set in negative_samples_array if ]
    #fre2 = sum([1 for active_set in positive_samples_array]
    n1, n2 = 0, 0
    for i in (positive_samples_array):
      new_array = i - self.active_array
      if (-1 in new_array):
        continue
      else:
        n1 += 1
    for j in negative_samples_array:
      new_array = j - self.active_array
      if (-1 in new_array):
        continue
      else:
        n2 += 1
    
    if ((n2/length2) > (n1/length1)):
      self.active_array = np.array([]) 
      return True
    else:
      return False
# 交叉节点
class Random_intersection_tree():
  pass

def pro_rif_pos(positive_samples_array, negative_samples_array, num_layer, branching, col_name, col_label):
  all_node = []
  layer_node = {}
  #layer_node= defaultdict(lambda:0)
  layer_node['0'] = []
  #x = []
  x0 = random.randint(0, len(positive_samples_array))
  #node = Node(positive_samples_array[x0])
  node = Node(None)
  node.active_array = positive_samples_array[x0]
  node.depth = 0
  real = True
  while (real):
    if (not node.stop(positive_samples_array, negative_samples_array)):
      real = False
      layer_node['0'].append(node)
      all_node.append(layer_node)
    else:
      x0 = random.randint(len(positive_samples_array))
      node = Node(None)
      node.active_array = positive_samples_array[x0]
      node.depth = 0

  n = 1
  while (n < num_layer):
    print(n)
    layer_node = {}
    layer_node['%d'%n] = []
    pre_layer_node = all_node[n-1]
    #subsets = []
    if (len(all_node) == 1):
      for i in range(branching):
        x1 = random.choice(positive_samples_array)
        pre_node = Node(pre_layer_node['0'][0])
        insected_array = pre_node.activate(x1)
        #node = Node(insected_array)
        if (not pre_node.stop(positive_samples_array, negative_samples_array)):
          layer_node['%d'%n].append(pre_node)
      all_node.append(layer_node)
      n += 1
    else:
      for j in pre_layer_node['%d'%(n-1)]:
        for i in range(branching):
          x1 = random.choice(positive_samples_array)
          #node = Node(x1)
          pre_node = Node(j)
          insected_array = pre_node.activate(x1)
          #node = Node(insected_array)
          if (not pre_node.stop(positive_samples_array, negative_samples_array)):
             layer_node['%d'%n].append(pre_node) # intersected node
            #x.append(x1)
      all_node.append(layer_node)
      n += 1

  num = 1
  #for value in all_node[-1]['8']:
  '''
  workbook = xlwt.Workbook(encoding='ascii')
  worksheet = workbook.add_sheet('leaf_node')
  worksheer.write(0, 0, '')
  for value in all_node[-1]['8']:
    worksheet.write(num, 0, num)
    worksheet.write(num, 1, value)
    num += 1   
  workbook.save('leaf_node_pos.xls')    
  '''
  print(all_node[-1])
  new_array = all_node[-1]['%d'%(n-1)][0].active_array.reshape(1, 22)
  #print(new_array)
  for i in all_node[-1]['%d'%(n-1)][1:]:
    new_array = np.insert(new_array, -1, i.active_array, axis=0)
  df_subsets = pd.DataFrame(new_array)
  df_subsets.columns = col_name[:]+col_label
  df_subsets.to_csv('leaf_node_pos.csv')
  print('Successful')
  
def pro_rif_neg(positive_samples_array, negative_samples_array, num_layer, branching, col_name, col_label):
  all_node = []
  layer_node = {'0':[]}
  #layer_node = defaultdict(int)
  layer_node['0'] = []
  x0 = random.randint(0, len(negative_samples_array))
  #node = Node(positive_samples_array[x0])
  node = Node(None)
  node.active_array = negative_samples_array[x0]
  node.depth = 0
  real = True
  while (real):
    if (not node.stop(positive_samples_array, negative_samples_array)):
      real = False
      layer_node['0'].append(node)
      all_node.append(layer_node)
    else:
      x0 = random.randint(len(negative_samples_array))
      node = Node(None)
      node.active_array = negative_samples_array[x0]
      node.depth = 0

  #layer_node['0'].append(node)
  n = 1
  while (n < num_layer):
    print(n)
    layer_node = {}
    layer_node['%d'%n] = []
    pre_layer_node = all_node[n-1]
    if (len(all_node) == 1):
      for i in range(branching):
        x1 = random.choice(negative_samples_array)
        pre_node = Node(pre_layer_node['0'][0])
        insected_array = pre_node.contrast(x1)
        #node = Node(insected_array)
        if (not node.stop(positive_samples_array, negative_samples_array)):
          layer_node['%d'%n].append(pre_node)
      all_node.append(layer_node)
    else:
      for j in pre_layer_node['%d'%(n-1)]:
        for k in range(branching):
          x1 = random.choice(negative_samples_array)
          pre_node = Node(j)
          insected_array = pre_node.contrast(x1)
          #node = Node(insected_array)
          if (not node.stop(positive_samples_array, negative_samples_array)):
            layer_node['%d'%n].append(pre_node)
      all_node.append(layer_node)
    n += 1
  
  new_array = all_node[-1]['%d'%(n-1)][0].active_array.reshape(1, 22)
  for i in all_node[-1]['%d'%(n-1)][1:]:
    new_array = np.insert(new_array, -1, i.active_array, axis=0)
  df_subsets = pd.DataFrame(new_array)
  df_subsets.columns = col_name[:]+col_label
  df_subsets.to_csv('leaf_node_neg.csv')
  print('successfully')
  #for value in all_node[-1]['8']:       
  '''
  num = 1       
  workbook = xlwt.Workbook(encoding='ascii')    
  worksheet = workbook.add_sheet('leaf_node')   
  worksheer.write(0, 0, '')                     
  for value in all_node[-1]['8']:               
    worksheet.write(num, 0, num)                
    worksheet.write(num, 1, value)              
    num += 1                                    
  workbook.save('leaf_node_neg.xls')             
  '''
if __name__ == '__main__':
  df = pd.read_csv('1st_2nd_onehot_prop_diff_dataset.csv')
  cols_name_features = []
  for i in list(df.columns[1:22]):
    cols_name_features.append(i)
  cols_name_label = [df.columns[-2]]
  df_positive = df.loc[df['D[cat] m2/s'] > 1.72e-11]
  df_negative =  df.loc[df['D[cat] m2/s'] <= 1.72e-11]
  df_positive['D[cat] m2/s'] = 1
  df_negative['D[cat] m2/s'] = 0
  #df_positive
  data_positive = np.array(df_positive[cols_name_features[:]+cols_name_label])
  data_negative = np.array(df_negative[cols_name_features[:]+cols_name_label])

  #pro_rif_pos(data_positive, data_negative, num_layer=9, branching=3, col_name=cols_name_features, col_label=cols_name_label)  
  pro_rif_neg(data_positive, data_negative, num_layer=9, branching=3, col_name=cols_name_features, col_label=cols_name_label)

