python main_eval_dncnn_origin.py --resume ./checkpoint/model_pools/model_best_dncnn_origin.pth --result ./eval_result/model_best_dncnn_origin/Gaussian/ --n_type 1 --cuda --input_number 1
python main_eval_dncnn_origin.py --resume ./checkpoint/model_pools/model_best_dncnn_origin.pth --result ./eval_result/model_best_dncnn_origin/random_impulse/ --n_type 2 --cuda --input_number 1
python main_eval_dncnn_origin.py --resume ./checkpoint/model_pools/model_best_dncnn_origin.pth --result ./eval_result/model_best_dncnn_origin/salt_and_pepper/ --n_type 3 --cuda --input_number 1
python main_eval_dncnn_origin.py --resume ./checkpoint/model_pools/model_best_dncnn_origin.pth --result ./eval_result/model_best_dncnn_origin/Poisson/ --n_type 4 --cuda --input_number 1

python main_eval_dncnn_origin.py --resume ./checkpoint/model_pools/model_epoch99_dncnn_origin.pth --result ./eval_result/model_epoch99_dncnn_origin/Gaussian/ --n_type 1 --cuda --input_number 1
python main_eval_dncnn_origin.py --resume ./checkpoint/model_pools/model_epoch99_dncnn_origin.pth --result ./eval_result/model_epoch99_dncnn_origin/random_impulse/ --n_type 2 --cuda --input_number 1
python main_eval_dncnn_origin.py --resume ./checkpoint/model_pools/model_epoch99_dncnn_origin.pth --result ./eval_result/model_epoch99_dncnn_origin/salt_and_pepper/ --n_type 3 --cuda --input_number 1
python main_eval_dncnn_origin.py --resume ./checkpoint/model_pools/model_epoch99_dncnn_origin.pth --result ./eval_result/model_epoch99_dncnn_origin/Poisson/ --n_type 4 --cuda --input_number 1
