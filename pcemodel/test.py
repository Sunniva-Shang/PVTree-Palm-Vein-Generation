import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util import util
import torch
import numpy as np
from tqdm import tqdm

# options
opt = TestOptions().parse()
opt.num_threads = 1   # test code only supports num_threads=1
opt.batch_size = 1   # test code only supports batch_size=1
opt.serial_batches = True  # no shuffle

# create dataset
dataset = create_dataset(opt)
model = create_model(opt)
model.setup(opt)
# model.eval()
print('Loading model %s' % opt.model)

def get_z_random2(batch_size, nz):
    num = batch_size
    base_vector = np.random.normal(loc=0, scale=1, size=(1, nz))
    base_vector /= np.linalg.norm(base_vector)  
   
    random_directions = np.random.normal(loc=0, scale=1, size=(num, nz))
    random_directions /= np.linalg.norm(random_directions, axis=1,keepdims=True)
    fixed_distance =0.5
    fixed_distance_vectors = base_vector + random_directions * fixed_distance
    z=torch.from_numpy(fixed_distance_vectors)
    return z

def saveimg(img, p, oripath, savepath):
    img = util.tensor2im(img)
    sp = os.path.dirname(p) 
    dp = sp.replace(oripath, savepath)
    os.makedirs(dp, exist_ok=True)
    util.save_image(img, p.replace(sp, dp))

if __name__ == '__main__':
    length = len(dataset)
    dataset = iter(dataset)
    for i in tqdm(range(0, length, opt.n_samples)):
        
        z_samples = get_z_random2(opt.n_samples + 1, opt.nz)
        
        images = []
        names = []
        for j in range(0, opt.n_samples):
            data = next(dataset)
            model.set_input(data)

            fake_B, p = model.test(z_samples[[j]], encode=False)
            # print(p)
            saveimg(fake_B, p[0], opt.dataroot, opt.results_dir)
    
