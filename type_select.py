import numpy as np
from collections import namedtuple
from data_helper.read_data import read_babi

def run_type_select(task_list,args, MainModel):
    #tasks = list(range(1,21))
    #tasks = list(range(1,4))
    args.action_num = len(task_list)
    
    ## data set ##
    print ('Reading babi data...')
    train, test, words, args.story_size, args.sentence_size, args.question_size = read_babi(task_list, args.batch_size, False,'seperate')
    print ('Reading babi over!')
    train_val = [data_set.split_dataset(args.val_ratio) for data_set in train]
    test_val = [data_set.split_dataset(args.val_ratio) for data_set in test]

    print("story size: {}".format(args.story_size))
    print("sentence size: {}".format(args.sentence_size))
    print("question size: {}".format(args.question_size))

    print ('training set size : ',[data.size() for data in train])
    print ('testing set size : ',[data.size() for data in test])
    ## create params ##
    params_dict = vars(args)
    params_class = namedtuple('params_class', params_dict.keys())
    params = params_class(**params_dict)
  
    main_model = MainModel(words,params)
    if params.action == 'train':
        main_model.train(train,train_val,'soft_q')
    elif params.action == 'test':
        if params.load_dir == '':
            print ('No loading DQN model')
        main_model.QA_reset()
        for i in range(100):
            #main_model.test(test,test_val,3,params.batch_size,i,'soft_q')
            main_model.test(test,test_val,40,20,i,'soft_q')
    
    '''
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
        main_model = MainModel(words, params)
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
    '''
