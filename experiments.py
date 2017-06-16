import os
import numpy as np
import json
import tensorflow as tf
from collections import namedtuple
from data_helper.read_data import read_babi

def load_params_dict(filename):
    with open(filename, 'r') as file:
        params_dict = json.load(file)
    return params_dict


def run_experiments(task, args, MainModel, RL):
    ## data set ##
    args.task = task
    train, test, words, args.story_size, args.sentence_size, args.question_size = read_babi(task, args.batch_size, False)
    val = train.split_dataset(args.val_ratio)
    print("training count: {}".format(train.count))
    print("testing count: {}".format(test.count))

    print("story size: {}".format(args.story_size))
    print("sentence size: {}".format(args.sentence_size))
    print("question size: {}".format(args.question_size))

    ## create params ##
    params_dict = vars(args)
    params_class = namedtuple('params_class', params_dict.keys())
    params = params_class(**params_dict)
    if not params.load_dir == '':
        raise Exception("Experiments Mode now only support training from scratch!")
    else:
        assert args.action == "train"
        if tf.gfile.Exists(args.save_dir):
            tf.gfile.DeleteRecursively(args.save_dir)
        os.makedirs(args.save_dir, exist_ok=True)

    if not params.expert_dir == '':
        params_filename = os.path.join(params.expert_dir, 'params.json')
        load_params = load_params_dict(params_filename)
        if not load_params['task'] == params.task:
            raise Exception("incompatible task with expert model!")
        if not load_params['target'] == 'expert':
            raise Exception("dir contains no expert model!")
        expert_params = params._replace(action='test', load_dir=params.expert_dir, **load_params)
    else:
        raise Exception("Need to load an expert from expert_dir to run experiments!")
    if not params.lm_dir == '':
        params_filename = os.path.join(params.lm_dir, 'params.json')
        load_params = load_params_dict(params_filename)
        if not load_params['target'] == 'lm':
            raise Exception("dir contains no language model!")
        lm_params = params._replace(action='test', 
                                    load_dir=params.lm_dir,
                                    lm_num_steps=1,
                                    lm_batch_size=params.batch_size,
                                    **load_params)
    else:
        if params.target == 'learner':
            print("No language model used in learner!")
        lm_params = None
    
    #record = np.array(['training sample','train acc','val acc','test acc'])
    #record = np.expand_dims(record, axis=0)
    num_sample = min(train.count, 256*10*len(task))
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
