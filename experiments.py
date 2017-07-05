import os
import numpy as np
import csv
import tensorflow as tf


def run_experiments(MainModel, params, expert_params, lm_params, words, train, val, RL):
    assert (params.action == 'experiments_rl' and RL) or (params.action == 'experiments_nonrl' and not RL)
    res_file = open(os.path.join(params.save_dir, 'result.txt'), 'w')
    res_writer = csv.writer(res_file)
    res_writer.writerow(['task_'+'_'.join([str(i) for i in params.task])])
    res_writer.writerow(['run', 'ratio', 'MA loss', 'MA acc', 'loss', 'acc'])
    num_sample = min(train.count, 256*10*len(params.task))
    train = train[:num_sample]
    print("train_num:", train.count)
    for pre_ratio in np.array(range(10, 100, 10))/100.:
        learner_params = params._replace(save_dir=os.path.join(params.save_dir, "pretrain_"+str(pre_ratio)))
        all_MA_val_loss = []
        all_MA_val_acc = []
        all_val_loss = []
        all_val_acc = []
        for run in range(5):
            if tf.gfile.Exists(learner_params.save_dir):
                tf.gfile.DeleteRecursively(learner_params.save_dir)
            os.makedirs(learner_params.save_dir, exist_ok=True)
            print('pre-train data ratio  : ', pre_ratio)
            pretrain_data = train[:int(train.count*pre_ratio)]
            print("pre_train_num:", pretrain_data.count)
            rltrain_data = train[int(train.count*pre_ratio):]
            print("rltrain_num:", rltrain_data.count)
            ## run action ##
            main_model = MainModel(words, learner_params, expert_params, lm_params)
            main_model.pre_train(pretrain_data, val, pretrain_data)
            main_model.set_params(num_epochs=256)
            if RL:
                main_model.rl_train(rltrain_data, val, pretrain_data, Q_limit=num_sample)
            MA_val_loss, MA_val_acc, val_loss, val_acc = main_model.QA_retrain(rltrain_data,
                                                                               val,
                                                                               pretrain_data,
                                                                               Q_limit=num_sample)
            res_writer.writerow([run+1, pre_ratio, MA_val_loss, MA_val_acc, val_loss, val_acc])
            all_MA_val_loss.append(MA_val_loss)
            all_MA_val_acc.append(MA_val_acc)
            all_val_loss.append(val_loss)
            all_val_acc.append(val_acc)
        res_writer.writerow(['avg',
                             pre_ratio,
                             np.mean(all_MA_val_loss),
                             np.mean(all_MA_val_acc),
                             np.mean(all_val_loss),
                             np.mean(all_val_acc)])
        #record = np.concatenate((record,tmp),axis=0)
    res_file.close()
    """
    if params.task == 'all':
        output_path = save_dir + '/record/all.csv'
    else :
        output_path = save_dir + '/record/%s.csv' % task[0]
    np.savetxt(output_path,record,fmt ='%s,%s,%s,%s' )
    print ('save record to ',output_path)
    """
