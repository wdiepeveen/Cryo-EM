Ellipsoidal Support Lifting for Cryo-EM joint 3D map reconstruction and rotation estimation
=========================================

Regularized global optimisation over Riemannian manifolds

        [1] W. Diepeveen, J. Lellmann, O. Öktem, C-B. Schönlieb.  
        Regularizing orientation estimation in cryogenic electron microscopy three-dimensional map refinement through measure-based lifting over riemannian manifolds.
        SIAM Journal on Imaging Sciences, 16(3):1440–1490, 2023.

Setup
-----

The recommended (and tested) setup is based on MacOS 11.4 running Python 3.6. Install the following dependencies with anaconda:

    # Create conda environment
    conda create --name esl1 python=3.6
    conda activate esl1

    # Clone source code and install
    git clone https://github.com/wdiepeveen/Cryo-EM.git
    cd "Cryo-EM"
    pip install -r requirements.txt


Reproducing the experiments in [1]
----------------------------------

The following jupyter notebooks have been used to produce the results in [1]. 
The tables and plots are directly generated after running the notebook. 

* 6.1. Asymptotic behaviour (Tab. 1 to 6):

        experiments/testing_asymptotics/experiment.ipynb

* 6.2. Joint 3D map reconstruction and rotation estimation (Fig. 4, 5 and 6):

        experiments/testing_joint_refinement/experiment.ipynb
