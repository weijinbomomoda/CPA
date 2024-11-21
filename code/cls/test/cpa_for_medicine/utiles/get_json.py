import json


json_path='../cfgs/superparam.json'

def json_data():
    jsontext = {}
    jsonpath = '/mnt/wjb/code/cls/mymodel/cfgs/superparam.json'
    jsontext['data_dir']= '/mnt/wjb/dataset/Project_CPA/work_dir/train'
    jsontext['csv_dir']='/mnt/wjb/dataset/Project_CPA/work_dir/info'
    jsontext['n_splits'] = 3
    jsontext['num_epochs']=50
    jsontext['fold']=0
    jsontext['mode']='cover'
    jsontext['test_mode'] = 'continue'
    # jsontext['resnet-layers']=10
    jsontext['lr']=1e-4
    jsontext['batch_size']=64
    jsontext['cls']=2
    jsontext['num_workers']=8
    jsontext['test_path']='/mnt/wjb/dataset/Project_CPA/work_dir/test'
    # print('/mnt/llz/media/imagenet/ILSVRC2012/')



    jsondata = json.dumps(jsontext, indent=4, separators=(',', ': '))
    with open(jsonpath, 'w') as f:
        f.write(jsondata)

def open_json():
    jsonpath = '/mnt/wjb/code/cls/mymodel/cfgs/superparam.json'
    with open(jsonpath,'r') as f:
        datadict=json.load(f)
    return datadict

def test():
    json_data()

if __name__ == '__main__':
    test()
