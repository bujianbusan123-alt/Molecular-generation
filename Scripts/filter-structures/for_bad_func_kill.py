from rdkit.Chem import Draw
from rdkit import Chem
import os
from rdkit.Chem import AllChem
import numpy as np
# D:\Anaconda\envs\python36\Library\share\RDKit\Data   FunctionalGroups.txt


datapath='F:/OnlinePacket/programfiles/python/GNNtest/GraphINVENT-master/output/output_7-19_LowT-secend-round/select'#smiles文件路径
filename='sum.smi'#目标smiles文件名
checksource='F:/onlinepacket/programfiles/python/gnntest/graphinvent-master/output/--------------data-------------/functionalization_bad.txt'#官能团目录
os.chdir(datapath)

data=np.loadtxt(filename, dtype=str, comments='**', skiprows=1, encoding='UTF-8')
fg=np.loadtxt(checksource,dtype=str,comments='//',encoding='UTF-8')
print(fg)

os.chdir(datapath)
if '.txt' in filename:
    dp = filename.replace('.txt', '_draws_after_kill_bad_func')
else:
    dp = filename.replace('.smi', '_draws_after_kill_bad_func')
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

smi = open("smiles-aftercheck.txt", 'w')

for i in range(0, len(data[:, 0])):
    if i==0:
        smi.write('SMILES Name')
        for fgn in fg[:,0]:
            smi.write(' '+fgn)
        smi.write('\n')
    try:
        mol=data[i][0]
        m = AllChem.AddHs(Chem.MolFromSmiles(mol))
        AllChem.EmbedMolecule(m)
        AllChem.MMFFOptimizeMolecule(m)
        light=True
        for fgs in fg[:,1]:
            patt = Chem.MolFromSmarts(fgs)
            if m.HasSubstructMatch(patt):
                light=False
        if light:
            smi.write(data[i][0] + ' ' + data[i][1] + '\n')
            draw = Draw.MolToImage(Chem.MolFromSmiles(mol))
            draw.save(data[i][1] + '.jpg')
            print(mol,1,'pass')
        else:
            print(mol,1,'fail')
    except:
        mol=data[i][0]
        print(mol,0,'dead---')
smi.close()

if os.path.exists('bad_FC_Draws'):
    del_list = os.listdir('bad_FC_Draws')
    for f in del_list:
        file_path = os.path.join(dp, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
else:
    os.mkdir('bad_FC_Draws')
ph2=os.path.abspath('bad_FC_Draws')
os.chdir(ph2)
print('---------------------------------------')
for i in range(0,len(fg[:,1])):
    try:
        mol=fg[i][1]
        m = Chem.MolFromSmarts(mol)
        print(mol,1)
        draw = Draw.MolToImage(m)
        draw.save(fg[i][0]+'.jpg')
    except:
        mol=fg[i][1]
        print(mol,0)
