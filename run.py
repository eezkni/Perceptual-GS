import os

device = 0
save_path = "/models/path"
dataset_path = '/datasets/path'

def run_exp(exp_name, datatsets, scenes, resolutions):
    for dataset in datatsets:

        for idx, scene in enumerate(scenes[dataset]):
            source_path = f"{dataset_path}/{dataset}/{scene}"
            model_path = f"{save_path}/{exp_name}/{dataset}/{scene}"
            resolution = resolutions[dataset][idx]

            # train
            train_cmd = f'CUDA_VISIBLE_DEVICES={device} python train.py -s {source_path} -m {model_path} --eval -r {resolution}'
            os.system(train_cmd)

            # render
            render_cmd = f'CUDA_VISIBLE_DEVICES={device} python render.py -s {source_path} -m {model_path}'
            os.system(render_cmd)
            
            # metric
            one_cmd = f'CUDA_VISIBLE_DEVICES={device} python metrics.py -m {model_path}'
            os.system(one_cmd)

if __name__ == "__main__":
    datatsets = ['blending', 'tandt', 'mipnerf360', 'bungeenerf']
    scenes = \
    {
        'mipnerf360':['bicycle', 'bonsai', 'counter', 'kitchen', 'room', 'stump', 'garden', 'flowers', 'treehill'],
        'blending':['drjohnson', 'playroom'],
        'tandt':['train', 'truck'],
        'bungeenerf':['amsterdam', 'barcelona', 'bilbao', 'chicago', 'hollywood', 'pompidou', 'quebec', 'rome'],
    }
    resolutions = \
    {
        'mipnerf360':[4, 2, 2, 2, 2, 4, 4, 4, 4],
        'blending':[1, 1],
        'tandt':[1, 1],
        'bungeenerf':[-1, -1, -1, -1, -1, -1, -1, -1],
    }

    run_exp("exp_name", datatsets, scenes, resolutions)