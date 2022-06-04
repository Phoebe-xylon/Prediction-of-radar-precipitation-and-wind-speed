#coding:utf8
import visdom
import time
import numpy as np

class Visualizer(object):
    '''
    封装了visdiom的基本操作，但仍然可以通过’self.vis.function‘
    或者’self.function‘调用原生的visdom接口
    例如
    self.text('hello visdom')
    self.histogram(torch.randn(1000))
    self.line(torch.arange(0,10),torch.arange(1,11))
    '''
    def __init__(self,env = 'default',**kwargs):
        self.vis = visdom.Visdom(env = env, **kwargs)
        self.index = {}
        self.log_text = ''

    def reinit(self,env = 'default',**kwargs):
        '''
        修改visdom的配置
        '''
        self.vis = visdom.Visdom()
        return self
    
    def plot_many(self,d):
        '''
        一次plot多个
        '''
        for k, v in d.items():
            self.plot(k,v)
    
    def img_many(self, d):
        for k, v in d.items():
            self.imag(k,v)

    def plot(self,name,y,**kwargs):
        '''
        self.plot('loss',1.00)
        '''
        x = self.index.get(name,0)
        self.vis.line(Y = np.array([y]),X = np.array([x]),
                win =(name),
                opts = dict(title = name ),
                update = None if x ==0 else 'append',
                **kwargs
                )
        self.index[name] = x + 1

    def scatter(self,name,y,**kwargs):
        '''
        self.plot('loss',1.00)
        '''
        self.vis.scatter(X = y[0,0],
                win =(name),
                opts = dict(title = name,markersize=1 ),
                **kwargs
                )

    def imag(self,name,img_,**kwargs):
        '''
        self.img('input_imgs',torch.Tensor(64,64))
        self.img('input_imgs',torch.Tensor(3,64,64))
        self.img('input_imgs',torch.Tensor(100,1,64,64))
        self.img('input_imgs',torch.Tensor(100,3,64,64),nrows = 10)
        !!!don`t~~self.img('input_imgs',torch.Tensor(100,64,64),nrows = 10 )~~!!!
        '''
        self.vis.images(img_.cpu().numpy(),
                win = (name),
                opts = dict(title = name),
                **kwargs
                )

    def log(self,info,win = 'log_text'):
        '''
        self.log({'loss':1,'lr':0.0001})
        '''
        self.log_text += ('[{time}] {info} <br>'.format(time = time.strftime('%m%d_%H%M%S'),info = info))
        self.vis.text(self.log_text,win)

    def __getattr__(self,name):
        '''
        自定义的plot，image，log，plot_many等除外
        self.function等价于self.vis.function
        '''
        return getattr(self.vis,name)
