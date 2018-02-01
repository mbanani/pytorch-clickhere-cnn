tensorboard_logdir = '/z/home/mbanani/tensorboard_logs'

root_dir                = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

render4cnn_weights      =  os.path.join(root_dir, 'model_weights/render4cnn.pth')
ft-render4cnn_weights   =  os.path.join(root_dir, 'model_weights/ryan_render.npy')
clickhere_weights       =  os.path.join(root_dir, 'model_weights/ch_cnn.npy')
