import torch
import time


class BasicModule(torch.nn.Module):
    '''
    封装了nn.Module，主要提供save和load两个方法
    '''
    def __init__(self):
        super(BasicModule,self).__init__()
        self.model_name = str(self)[:-2] # 模型的默认名字

    def load(self,path,map_location):
        '''
        可加载指定路径的模型
        '''
        self.load_state_dict(torch.load(path,map_location = map_location))

    def save(self,name = None):
        '''
        保存模型，默认使用“模型名字+时间”作为文件名，
        如AENet_0710_23:57:29.pth
        '''
        if name is None :
            prefix = '../user_data/'+self.model_name+'_'
            name = time.strftime(prefix+'%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(),name)
        return name


