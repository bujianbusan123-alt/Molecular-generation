# Molecular-generation
<div align='center' >Artificial Intelligence-Assisted Electrolyte Design to Improve Li+ Diffusion</div>
<div align=center><img decoding="async" src="procedure.bmp" width="60%"></div>
## Introduction
The problem of electrolyte freezing and power loss on lithium batteries under low-temperature conditions requires immediate attention, especially applications to high altitude,
high latitude regions and aerospace have been limited. Enhancing the Li<sup>+</sup> diffusion coefficient represents a crucial direction for improvement. However,
improving Li<sup>+</sup> diffusion coefficient in existed electrolytes has proven to be challenging, exploring new solvents or additives in electrolyte may have great potential. 
In this study, we present a novel strategy that utilizes advanced techniques to design and find five novel molecules as additives with high Li<sup>+</sup> diffusion coefficient. 
Our methodology involves two rounds of Molecular generation (MG) and Molecular dynamics (MD), and three rounds of Machine learning (ML), resulting in the highest Li<sup>+</sup> diffusion 
coefficient of the generated additive being 3.96 times that of the benchmark system (1.72×10<sup>-11</sup> m/s<sup>2</sup>). These findings are of great significance in improving Li<sup>+</sup> 
diffusion coefficient in practical applications and addressing this persistent issue. Furthermore, these strategies and results provide valuable insights for further investigation into other
challenges of lithium batteries. 
## Dependencies
rdkit==2022.03.3<br />
scipy==1.8.0<br />
numpy==1.21.5<br />
pandas==1.4.1<br />
scikit-learn==1.0.2<br />
xgboost==1.5.0<br />
shap==0.41.0<br />
moses
# Installation
conda create --name version python=3.8
# Generative model GraphInvent
1. git clone https://github.com/MolecularAI/GraphINVENT.git,<br />
2. pretraining and training the generative model using collected dataset, <br />
3. generated plenty of novel molecules, <br />
4. filtering the invalid and repeated molecules using scripts.
# Machine learning methods
Classification models were trained by Scikit-learn except XGBoost,<br />
the features for machine learning models are obatined by python get_functional_group_features.py,<br />
you can also define some important physical descriptors on the bulk and interface.
# Synthesizable analysis
## SA
A heuristic estimate of how hard (10) or how easy (1). it is to synthesize a given molecule. SA score is based on a combination of the molecule’s fragments contributions.
And SA score threshold change from 6.0 to -4.5, SA score would yield similar results as SYBA, here, we set the threshold of SA score 4.5. SA was assembled in MOSES https://github.com/molecularsets/moses
## SYBA
SYBA is a fragment-based method for dividing the organic compounds into easy-synthesize (ES) or hard-to-synthesize (HS).
It is based on their frequencies in the dataset of ES and HS. It is based on a Bernoulli naïve Bayes classifier that is used to assign SYBA score contributions to individual
fragment based on the frequencies in the database of ES and HS. SYBA was trained previously on ES molecules in ZINC15 database and on HS molecules generated by
the Nonpher methodology. And SYBA is publicly available at https//github.com/lich-uct/syba. 
Here, we set the threshold of SYBA score -18.6. <br />

