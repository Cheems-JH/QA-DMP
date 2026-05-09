Code:
- GMM_DMP.py:               >> the code of the basic DMP driven by GMM, used for comparison and intermediate data processing.
- QA_DMP_for_2D_demos.py    >> the code of QA-DMP learning from 2D demos and generate outputs.
- QA_DMP_for_3D_demos.py    >> the code of QA-DMP learning from 3D demos and generate outputs.
- requirements.yml          >> the required Conda environment and library dependencies.

Dataset:
- 2D_simulation_demos.npy   >> the 40 multi-quality demos (x,y) with length 1021 obtained in the 2D simulation.
- 3D_simulation_demos.npy   >> the 40 multi-quality demos (x,y,z) with length 1087 obtained in the 3D simulation.
- pen_draw_demos_30.npy     >> the 30 multi-quality demos (x,y) with length 890 obtained in the pen-drawing experiment.
- pen_draw_demos_80.npy     >> the 80 multi-quality demos (x,y) with length 890 obtained in the pen-drawing experiment,
                               and the first 30 demos are the same as those in the file [pen_draw_demos_30.npy].
- pick_place_demos_thy.pny  >> the 30 multi-quality demos (\theta_y) with length 2500 obtained in the pick-and-place experiment.
- pick_place_demos_xyz.pny  >> the 30 multi-quality demos (x,y,z) with length 2500 obtained in the pick-and-place experiment,
                               together with the previous file, it forms the complete demonstration data for this experiment.

Notes:
- The first row of each .npy dataset serves as the corresponding baseline, it is excluded from training and used solely for comparative analysis.
- Learning for higher-dimensional demonstrations can be easily extended from QA_DMP_for_2D_demos.py or QA_DMP_for_3D_demos.py,
  as both DMP and QA-DMP perform training and learning independently for each dimension of the demonstrations.
- QA-DMP is less sensitive to hyperparameters, delivering much better results than standard DMP over a large range of settings,
  this flexibility facilitates easier adjustments for different tasks. Also, the code demonstrates stable performance without requiring specific random seeds.
- The implementation of TS2Vec directly uses its official source code: https://github.com/zhihanyue/ts2vec.