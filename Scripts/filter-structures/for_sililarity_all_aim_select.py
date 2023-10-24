from rdkit.Chem import Draw
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem.AtomPairs import Torsions
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from rdkit.Chem import AllChem
import numpy as np
from rdkit import DataStructs
# --------------------------------------------------------------------------------初始参数设定
selectpath = 'F:/onlinepacket/programfiles/python/gnntest/graphinvent-master/output/output_7-19_lowt-secend-round/select'
filename = 'sum.smi'
checkname='checkaim.txt'#没有就填：‘None’
selectnum=170#综合最优筛选数量
single_select_num=20#单分子目标筛选数量
D_standard=3.20E-11#非零值
fpskinds_use=4#使用四种分子指纹
endc=0.75#末端重要性
# --------------------------------------------------------------------------------
os.chdir(selectpath)
data = np.loadtxt(filename, dtype=str, comments='**', skiprows=1, encoding='UTF-8')
try:
    checkdata = np.loadtxt(checkname, dtype=str, comments='**', skiprows=1, encoding='UTF-8')
    Ddata = np.zeros((len(checkdata[:, 2]), 1), dtype=float)
    for dd in range(len(checkdata[:, 2])):
        print(dd, checkdata[dd][2])
        Ddata[dd] = eval(checkdata[dd][2])
except:
    Ddata = np.zeros((len(checkdata[:, 2]), 1), dtype=float)

if '.txt' in filename:
    dp = filename.replace('.txt', '_similar_draws_'+str(selectnum)+'select'+'_endc='+str(endc))
else:
    dp = filename.replace('.smi', '_similar_draws_'+str(selectnum)+'select'+'_endc='+str(endc))
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
abs_dp=os.path.abspath(dp)

# --------------------------------------------------------------------------------指纹函数简化
# ------拓扑指纹 Chem.RDKFingerprint(x)
def fps0(mol):
    mfps = Chem.RDKFingerprint(mol)
    return mfps


# ------MACCS 指纹MACCSkeys.GenMACCSKeys(mol)
def fps1(mol):
    mfps = MACCSkeys.GenMACCSKeys(mol)
    return mfps


# ------Atom Pairs
def fps2(mol):
    mfps = Pairs.GetHashedAtomPairFingerprint(mol)
    return mfps


# topological torsions
def fps3(mol):
    mfps = Torsions.GetTopologicalTorsionFingerprint(mol)
    return mfps


# 摩根指纹（圆圈指纹）AllChem.GetMorganFingerprint(mol,2),摩根指纹又称为圆圈指纹。 产生摩根指纹的时候，需要指定指纹的半径。
def fps4(mol):
    mfps = AllChem.GetMorganFingerprint(mol, 2)# ECFP4
    return mfps


def fps5(mol):
    mfps = AllChem.GetMorganFingerprint(mol, 2, useFeatures=True)# FCFP4
    return mfps

fps_fun=[fps0,fps1,fps2,fps4]
sim_fun=[
    DataStructs.TanimotoSimilarity,
         # DataStructs.DiceSimilarity,
         # DataStructs.CosineSimilarity,
         # DataStructs.SokalSimilarity,
         # DataStructs.RusselSimilarity,
         # DataStructs.KulczynskiSimilarity,
         # DataStructs.McConnaugheySimilarity,
         # DataStructs.TverskySimilarity,
         # DataStructs.AllBitSimilarity,
         # DataStructs.AsymmetricSimilarity,
         # DataStructs.BraunBlanquetSimilarity,
         # DataStructs.RogotGoldbergSimilarity
]
sim_fun_bulk=[
    DataStructs.BulkTanimotoSimilarity,
         # DataStructs.BulkDiceSimilarity,
         # DataStructs.BulkCosineSimilarity,
         # DataStructs.BulkSokalSimilarity,
         # DataStructs.BulkRusselSimilarity,
         # DataStructs.BulkKulczynskiSimilarity,
         # DataStructs.BulkMcConnaugheySimilarity,
         # DataStructs.BulkTverskySimilarity,
         # DataStructs.BulkAllBitSimilarity,
         # DataStructs.BulkAsymmetricSimilarity,
         # DataStructs.BulkBraunBlanquetSimilarity,
         # DataStructs.BulkRogotGoldbergSimilarity
              ]
def smicor(m):#smiles标准化函数
    try:
        mol=Chem.MolToSmiles(Chem.MolFromSmiles(m))
    except:
        mol=Chem.MolToSmiles(Chem.MolFromSmarts(m))
    return mol
# -----------------------------------------------------------------------------------------------------------------
#字符串标准化
for i in range(0,len(data[:,0])):
    data[i][0]=smicor(data[i][0])
for i in range(0,len(checkdata[:,0])):
    checkdata[i][0]=smicor(checkdata[i][0])
# -----------------------------------------------------------------------------------------------------------------

score_sum=np.zeros(len(data[:,0]))
for iaim in range(0,len(checkdata[:,0])):
    if Ddata[iaim]>=D_standard:
        mn=checkdata[iaim][0]
        mol = Chem.MolFromSmiles(mn)


        # fps_1 = Chem.RDKFingerprint(mol)
        # fps_2 = MACCSkeys.GenMACCSKeys(mol)
        # fps_3 = Pairs.GetHashedAtomPairFingerprint(mol)
        # fps_4 = Torsions.GetTopologicalTorsionFingerprint(mol)
        # fps_5 = AllChem.GetMorganFingerprint(mol, 2)
        # fps_6 = AllChem.GetMorganFingerprint(mol, 2, useFeatures=True)
        # fps_list=[fps_1,fps_2,fps_3,fps_4,fps_5,fps_6]
        fps_list=[]
        for i in range(0,len(fps_fun)):
            fps_list.append(fps_fun[i](mol))

        var_list=np.zeros((len(fps_list),len(sim_fun)),dtype=float)

        # -----------------------------------------------------------------------------------------------------------------
        figp=str(iaim)+'_violinfig_'+checkdata[iaim][1]
        if os.path.exists(figp):
            absp=os.path.abspath(figp)
            del_list = os.listdir(figp)
            for f in del_list:
                file_path = os.path.join(absp, f)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            # pass
        else:
            os.mkdir(figp)
        os.chdir(figp)



        for i in range(0,len(fps_list)):
            fps_con=fps_list[i]
            fps_checklist=[]
            ii=0
            for mms in data[:,0]:
                datamol=Chem.MolFromSmiles(mms)
                fps_checklist.append(fps_fun[i](datamol))
            for simfun in sim_fun_bulk:
                try:
                    siml_list=simfun(fps_con,fps_checklist)
                    siml_var=np.var(siml_list, axis=None, dtype=None, out=None, ddof=0, keepdims=np._NoValue)
                    plt.figure()
                    # plt.axis([-1, 1, -1, 1])
                    plt.ylabel('('+str(i+1)+','+str(ii+1)+')'+'Similarity')
                    plt.title('var='+str(siml_var))
                    p1 = sns.violinplot(y=siml_list)
                    plt.savefig(str(i+1)+'号指纹-'+str(ii+1)+'号相似度计算方法' + '.jpg')
                    # plt.show()
                    # print(i+1,'号指纹',ii+1,'号相似度计算方法',len(siml_list),siml_var)
                    var_list[i][ii]=siml_var
                    draw = Draw.MolToImage(mol)
                    draw.save(checkdata[iaim][1] + '.jpg')
                except:
                    pass
                ii=ii+1
        var_max=np.max(var_list,axis=1)
        var_max=np.sort(var_max)[::-1]#降序
        print(var_max)
        f_s_pare=np.zeros((len(var_max),2))
        ci=0
        ci_max=fpskinds_use#取四种指纹
        score=np.zeros(len(data[:,0]))
        #--------------------单轮打分
        for varm in var_max:
            if ci==ci_max:
                break
            fi=np.argwhere(var_list==varm)[0][0]#指纹序号
            fi=int(fi)
            si=np.argwhere(var_list==varm)[0][1]#方法序号
            si=int(si)
            datafps_list=[]
            print(fi+1,si+1)
            for ii in range(0,len(data[:,0])):
                datamol = Chem.MolFromSmiles(data[ii][0])
                datafps = fps_fun[fi](datamol)
                datafps_list.append(datafps)
            scr_or=sim_fun_bulk[si](fps_list[fi],datafps_list)
            # print(scr_or)
            scr_0to1=preprocessing.minmax_scale(scr_or)
            # print(scr_0to1)
            score=scr_0to1*(1/ci_max)+score#分数序列
            ci=ci+1
        # --------------------单轮分子筛选
        plt.figure()
        plt.ylabel(checkdata[iaim][1] + 'Similarity')
        score_var=np.var(score, axis=None, dtype=None, out=None, ddof=0, keepdims=np._NoValue)
        plt.title('var=' + str(score_var))
        p1 = sns.violinplot(y=score)
        plt.savefig(checkdata[iaim][1] + 'Similarity' + '.jpg')

        smi = open("smiles.smi", 'w')
        smi.write('SMILES Name\n')
        scr_sort = np.sort(score)[::-1]  # 降序
        scr_rand = np.zeros((len(scr_sort), 1), dtype=int)
        for i in range(0, len(scr_sort)):
            rand = np.argwhere(score == scr_sort[i])[0][0]
            scr_rand[i] = int(rand)
        aim = single_select_num  # 选取前20个
        drawnum = 0
        for i in range(0, len(scr_rand[:, 0])):
            ms = data[scr_rand[i][0]][0]
            mol = Chem.MolFromSmiles(ms)
            scr_s = scr_sort[i]
            if drawnum < aim and ms not in checkdata[:, 0]:
                draw = Draw.MolToImage(mol)
                draw.save(data[scr_rand[i][0]][1] + '.jpg')
                # print(i, ms, data[i][1], 1, scr_s)
                smi.write(ms + ' ' + data[scr_rand[i][0]][1] + '\n')
                drawnum = drawnum + 1
            else:
                pass
        smi.close()
        print('此轮最高分',np.max(score),'此轮均分',np.average(score))

        score_sum=score_sum+score*(endc+(1-endc)*(Ddata[iaim]-D_standard)/(max(Ddata)-D_standard))#总分
        print(endc+(1-endc)*(Ddata[iaim]-D_standard)/(max(Ddata)-D_standard))


        os.chdir(os.path.join(selectpath, dp))

# -----------------------------------------------------------------------------------------------------------------
mdp='Molecular_draws_'+str(selectnum)+'select'
if os.path.exists(mdp):
    del_list = os.listdir(mdp)
    for f in del_list:
        file_path = os.path.join(mdp, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
    # pass
else:
    os.mkdir(mdp)
os.chdir(mdp)


#按排名提取
smi = open("smiles.smi", 'w')
smi.write('SMILES Name\n')
scr_record = open("scores.txt", 'w')
scr_record.write('SMILES Name score rand New_y(1)&n(0)\n')
scr_sort=np.sort(score_sum)[::-1]#降序
scr_rand=np.zeros((len(scr_sort),1),dtype=int)
for i in range(0,len(scr_sort)):
    rand=np.argwhere(score_sum==scr_sort[i])[0][0]
    scr_rand[i]=int(rand)
    # if scr_rand[i]==0:
    #     print('有0')
#百分比选取
# aim=0.5#选取前百分之五或其他
# scr_aim=scr_sort[int(aim*len(score))]
#固定数量选取
aim=selectnum#选取数量
# scr_aim=scr_sort[aim-1]
drawnum=0
for i in range(0, len(scr_rand[:, 0])):
    ms = data[scr_rand[i][0]][0]
    mol = Chem.MolFromSmiles(ms)
    scr_s=scr_sort[i]
    if drawnum<aim and ms not in checkdata[:,0]:
        draw = Draw.MolToImage(mol)
        draw.save(data[scr_rand[i][0]][1] + '.jpg')
        print(i, ms, data[scr_rand[i][0]][1], 1, scr_s)
        scr_record.write(ms + ' ' + data[scr_rand[i][0]][1] + ' ' + str(i + 1) + ' ' + str(scr_s) + ' ' + '1' + '\n')
        smi.write(ms + ' ' + data[scr_rand[i][0]][1] + '\n')
        drawnum=drawnum+1
    else:
        print(i, ms, data[scr_rand[i][0]][1], 0, scr_s)
        scr_record.write(ms + ' ' + data[scr_rand[i][0]][1] + ' ' + str(i + 1) + ' ' + str(scr_s) + ' ' + '0' + '\n')
smi.close()
scr_record.close()
plt.figure()
# plt.axis([-1, 1, -1, 1])
plt.ylabel('Synthesis Scores')
score_sum_var=np.var(score_sum, axis=None, dtype=None, out=None, ddof=0, keepdims=np._NoValue)
plt.title('var=' + str(score_sum_var))
p1 = sns.violinplot(y=score_sum)
plt.savefig('score_sum' + '.jpg')
