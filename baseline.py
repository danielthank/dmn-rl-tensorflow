import numpy as np
from collections import namedtuple
from data_helper.read_data import read_babi

def run_baseline(MainModel, params, expert_params, lm_params, words, train, val):
    assert params.action == 'baseline'
    record = np.array(['training sample','train acc','val acc','test acc'])
    record = np.expand_dims(record,axis=0)
    max_sample = min(train.count+1,2001)
    # max_sample = 201
    for training_sample in range(100,max_sample,100):
        tmp = np.zeros(4,dtype='float32')
        tmp[0] = training_sample
        print ('training data sample  : ',training_sample)
        train_sub = train[:training_sample]
        ## run action ##
        main_model = MainModel(words, params, expert_params, lm_params)
        main_model.pre_train(train_sub, val)
        #main_model.save_params()

        main_model.eval(test, name='Test')
        #main_model.decode(test, sys.stdout, sys.stdin, all=False)
        tmp[1] = main_model.train_acc
        tmp[2] = main_model.val_acc
        tmp[3] = main_model.test_acc
        tmp = np.expand_dims(tmp,axis = 0).astype('str')
        record = np.concatenate((record,tmp),axis=0)
    if params.task == 'all':
        output_path = save_dir + '/record/all.csv'
    else :
        output_path = save_dir + '/record/%s.csv' % task[0]
    np.savetxt(output_path,record,fmt ='%s,%s,%s,%s' )
    print ('save record to ',output_path)
