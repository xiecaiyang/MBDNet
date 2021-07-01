python main_eval_dncnn.py --resume ./checkpoint/model_pools/model_best_dncnn.pth --result ./eval_result/model_best_dncnn/Gaussian/ --n_type 1 --cuda
python main_eval_dncnn.py --resume ./checkpoint/model_pools/model_best_dncnn.pth --result ./eval_result/model_best_dncnn/random_impulse/ --n_type 2 --cuda
python main_eval_dncnn.py --resume ./checkpoint/model_pools/model_best_dncnn.pth --result ./eval_result/model_best_dncnn/salt_and_pepper/ --n_type 3 --cuda
python main_eval_dncnn.py --resume ./checkpoint/model_pools/model_best_dncnn.pth --result ./eval_result/model_best_dncnn/Poisson/ --n_type 4 --cuda

python main_eval_dncnn.py --resume ./checkpoint/model_pools/model_epoch99_dncnn.pth --result ./eval_result/model_epoch99_dncnn/Gaussian/ --n_type 1 --cuda
python main_eval_dncnn.py --resume ./checkpoint/model_pools/model_epoch99_dncnn.pth --result ./eval_result/model_epoch99_dncnn/random_impulse/ --n_type 2 --cuda
python main_eval_dncnn.py --resume ./checkpoint/model_pools/model_epoch99_dncnn.pth --result ./eval_result/model_epoch99_dncnn/salt_and_pepper/ --n_type 3 --cuda
python main_eval_dncnn.py --resume ./checkpoint/model_pools/model_epoch99_dncnn.pth --result ./eval_result/model_epoch99_dncnn/Poisson/ --n_type 4 --cuda

python main_eval_dncnn.py --resume ./checkpoint/model_pools/model_best_dncnn_est.pth --result ./eval_result/model_best_dncnn_est/Gaussian/ --n_type 1 --cuda --est_model
python main_eval_dncnn.py --resume ./checkpoint/model_pools/model_best_dncnn_est.pth --result ./eval_result/model_best_dncnn_est/random_impulse/ --n_type 2 --cuda --est_model
python main_eval_dncnn.py --resume ./checkpoint/model_pools/model_best_dncnn_est.pth --result ./eval_result/model_best_dncnn_est/salt_and_pepper/ --n_type 3 --cuda --est_model
python main_eval_dncnn.py --resume ./checkpoint/model_pools/model_best_dncnn_est.pth --result ./eval_result/model_best_dncnn_est/Poisson/ --n_type 4 --cuda --est_model

python main_eval_dncnn.py --resume ./checkpoint/model_pools/model_epoch99_dncnn_est.pth --result ./eval_result/model_epoch99_dncnn_est/Gaussian/ --n_type 1 --cuda --est_model
python main_eval_dncnn.py --resume ./checkpoint/model_pools/model_epoch99_dncnn_est.pth --result ./eval_result/model_epoch99_dncnn_est/random_impulse/ --n_type 2 --cuda --est_model
python main_eval_dncnn.py --resume ./checkpoint/model_pools/model_epoch99_dncnn_est.pth --result ./eval_result/model_epoch99_dncnn_est/salt_and_pepper/ --n_type 3 --cuda --est_model
python main_eval_dncnn.py --resume ./checkpoint/model_pools/model_epoch99_dncnn_est.pth --result ./eval_result/model_epoch99_dncnn_est/Poisson/ --n_type 4 --cuda --est_model
