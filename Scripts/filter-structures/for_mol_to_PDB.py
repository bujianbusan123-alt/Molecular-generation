import numpy as np
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

selectpath = 'F:/onlinepacket/programfiles/python/gnntest/graphinvent-master/output/output_7-19_lowt-secend-round/select/final_select'
filename = 'smiles.smi'

os.chdir(selectpath)
data=np.loadtxt(filename, dtype=str, comments='**', skiprows=1, encoding='UTF-8')
if '.txt' in filename:
    dp = filename.replace('.txt', '_pdb')
else:
    dp = filename.replace('.smi', '_pdb')
if os.path.exists(dp):
    del_list = os.listdir(dp)
    for f in del_list:
        file_path = os.path.join(dp, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
    # pass
else:
    os.mkdir(dp)
os.chdir(dp)

for i in range(0,len(data[:,0])):
    ms=data[i][0]
    mol= AllChem.AddHs(Chem.MolFromSmiles(ms))
    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol)
    Chem.MolToPDBFile(mol, data[i][0]+'.pdb')