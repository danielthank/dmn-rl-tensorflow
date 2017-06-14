import numpy as np
from collections import namedtuple
from data_helper.read_data import read_babi

def run_type_select(args, MainModel):
    #tasks = [[i] for i in range(1,21)]
    tasks = [[1]]
    words=None
    train_all = []
    test_all = []
    val_all = []
    max_story_size = 0
    max_sentence_size = 0
    max_question_size = 0
    ## data set ##
    print ('Reading babi data...')
    for task in tasks :
        print (task,end='\r')
        train, test, words, story_size, sentence_size, question_size = read_babi(task, args.batch_size, False,words)
        val = train.split_dataset(args.val_ratio)
        
        train_all.append(train)
        val_all.append(val)
        test_all.append(test)

        max_story_size = max(max_story_size,story_size)
        max_sentence_size = max(max_sentence_size,sentence_size)
        max_question_size = max(max_question_size, question_size)
    args.story_size = max_story_size
    args.sentence_size = max_sentence_size
    args.question_size = max_question_size
    
    #print("training count: {}".format(train.count))
    #print("testing count: {}".format(test.count))

    print("story size: {}".format(args.story_size))
    print("sentence size: {}".format(args.sentence_size))
    print("question size: {}".format(args.question_size))

    ## create params ##
    params_dict = vars(args)
    params_class = namedtuple('params_class', params_dict.keys())
    params = params_class(**params_dict)
  
    main_model = MainModel(words,params)
    main_model.train(train_all,val_all)
    
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
