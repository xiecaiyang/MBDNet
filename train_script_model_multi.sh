#python main_model_dncnn.py --ck ./checkpoint/model_dncnn --result ./result/model_dncnn --cuda
#python main_model_multi.py --ck ./checkpoint/model_multi/filter_number_320 --result ./result/model_multi/filter_number_320 --cuda
#python main_model_multi.py --ck ./checkpoint/model_multi/filter_number_320 --result ./result/model_multi/filter_number_320 --resume ./checkpoint/model_multi/model_epoch_85.pth --cuda
python main_model_multi.py --ck ./checkpoint/model_multi/filter_number_128 --result ./result/model_multi/filter_number_128 --cuda
