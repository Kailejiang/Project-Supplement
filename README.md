For an introductory tutorial of SchNet, see https://github.com/atomistic-machine-learning/schnetpack/blob/master/examples/tutorials/tutorial_01_preparing_data.ipynb



* log2npz.py :
  Taking the Gaussian output file as an example, read the energy and coordinate information and make it into .npz.



* npz2db.py :
  Convert npz files to db files for SchNet training.



* schnet_IIA.py :
  Include interatomic interaction attention and SMU blocks (replace it with the original schnetpack/representation/schnet.py when using it).



* train_with_trans_embedding.py :
  A method for using continuous feature transfer learning.


* Charge number coded.py :
  Encoding matrix of atomic features during embedding.
