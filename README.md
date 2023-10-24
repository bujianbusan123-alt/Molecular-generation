# Molecular-generation
Artificial Intelligence-Assisted Electrolyte Design to Improve Li<sup>+</sup> Diffusion<br />
<img decoding="async" src="procedure.bmp" width="50%">
## Introduction
The problem of electrolyte freezing and power loss on lithium batteries under low-temperature conditions requires immediate attention, especially applications to high altitude, high latitude regions and aerospace have been limited. Enhancing the Li<sup>+</sup> diffusion coefficient represents a crucial direction for improvement. However, improving Li<sup>+</sup> diffusion coefficient in existed electrolytes has proven to be challenging, exploring new solvents or additives in electrolyte may have great potential. In this study, we present a novel strategy that utilizes advanced techniques to design and find five novel molecules as additives with high Li<sup>+</sup> diffusion coefficient. Our methodology involves two rounds of Molecular generation (MG) and Molecular dynamics (MD), and three rounds of Machine learning (ML), resulting in the highest Li<sup>+</sup> diffusion coefficient of the generated additive being 3.96 times that of the benchmark system (1.72×10<sup>-11</sup> m/s<sup>2</sup>). These findings are of great significance in improving Li<sup>+</sup> diffusion coefficient in practical applications and addressing this persistent issue. Furthermore, these strategies and results provide valuable insights for further investigation into other challenges of lithium batteries. 
## Dependencies
rdkit==2022.03.3<br />
scipy==1.8.0<br />
numpy==1.21.5<br />
pandas==1.4.1<br />
scikit-learn==1.0.2<br />
xgboost==1.5.0<br />
shap==0.41.0
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
