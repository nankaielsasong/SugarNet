import torch
from torchviz import make_dot
import hiddenlayer as h

def visualize_model(model, save_name):
    x = torch.rand(8, 3, 256, 512)
    '''
    y = model(x)
    g = make_dot(y)
    g.render(save_name, view=False)'''
    vis_graph = h.build_graph(model, x)
    vis_graph.theme = h.graph.THEMES['blue'].copy()
    vis_graph.save(save_name)
    print('successfully print the structure of model')
