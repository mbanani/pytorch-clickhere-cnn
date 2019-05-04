import os

pascal3d_root           = '/home/mbanani/datasets/pascal3d'

root_dir                = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
render4cnn_weights      =  os.path.join(root_dir, 'model_weights/r4cnn.pkl')
ft_render4cnn_weights   =  os.path.join(root_dir, 'model_weights/ryan_render.npy')
clickhere_weights       =  os.path.join(root_dir, 'model_weights/ch_cnn.npy')
