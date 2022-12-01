import warnings
#初始化参数
class DefaultConfig(object):
    env = 'PSP'
    model = 'DNet'
    train_root = r'../data/Train/{}/'
    val_root = r'../data/TestA/{}/'
    train_csv = r'../data/Train.csv'
    val_csv = r'../data/TestA.csv'
    testpath = r'./submit/{}/'
    load_model_path = ""
    load_G_model_path = ""
    load_model_path = "../user_data/Dradar.pth"
    load_G_model_path = "../user_data/Gradar.pth"
    savepath = '../submit/{}/{}/'
    datatype = 'radar'
    factor = 10  #radar 70  wind  35  precip  10
    batch_size = 1
    use_gpu = False
    num_workers = 0
    print_freq = 12
    debug_file = '/tmp/debug'
    result_file = 'result.csv'
    max_epoch = 10
    lr = 0.001
    lr_decay = 0.5
    weight_decay = 1e-4
    phase = 'train'
    pattern = 'frame'
    framenum = 20
    print_log = True
    val = False
    model_save = True

    def parse(self,kwargs):
        '''
        根据字典kwargs更新config参数
        '''
        for k, v in kwargs.items():
            if not hasattr(self,k):
                warnings.warn("Warning: opt has not attribut %s" %k)
            setattr(self, k, v)
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k,getattr(self,k))


