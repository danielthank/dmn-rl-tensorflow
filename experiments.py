import os
import numpy as np
import json
import tensorflow as tf
from data_helper.read_data import read_babi


def run_experiments(MainModel, params, expert_params, lm_params, words, train, val, RL):
    assert (params.action == 'experiments_rl' and RL) or (params.action == 'experiments_nonrl' and not RL)
    #record = np.array(['training sample','train acc','val acc','test acc'])
    #record = np.expand_dims(record, axis=0)
    num_sample = min(train.count, 256*10*len(params.task))
    train = train[:num_sample]
    print("train_num:", train.count)
    for pre_ratio in np.array(range(10, 100, 10))/100.:
        learner_params = params._replace(save_dir=os.path.join(params.save_dir, "pretrain_"+str(pre_ratio)))
        #tmp = np.zeros(4, dtype='float32')
        #tmp[0] = training_sample
        print ('pre-train data ratio  : ', pre_ratio)
        pretrain_data = train[:int(train.count*pre_ratio)]
        print("pre_train_num:", pretrain_data.count)
        rltrain_data = train[int(train.count*pre_ratio):]
        print("rltrain_num:", rltrain_data.count)
        ## run action ##
        main_model = MainModel(words, learner_params, expert_params, lm_params)
        main_model.pre_train(pretrain_data, val, pretrain_data)
        if RL:
            main_model.rl_train(rltrain_data, val, pretrain_data, Q_limit=num_sample)
        val_loss, val_acc = main_model.QA_retrain(rltrain_data, val, pretrain_data, Q_limit=num_sample)

        #tmp[1] = main_model.train_acc
        #tmp[2] = main_model.val_acc
        #tmp[3] = main_model.test_acc
        #tmp = np.expand_dims(tmp,axis = 0).astype('str')
        #record = np.concatenate((record,tmp),axis=0)
    """
    if params.task == 'all':
        output_path = save_dir + '/record/all.csv'
    else :
        output_path = save_dir + '/record/%s.csv' % task[0]
    np.savetxt(output_path,record,fmt ='%s,%s,%s,%s' )
    print ('save record to ',output_path)
    """
