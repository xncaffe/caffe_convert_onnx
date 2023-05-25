import numpy as np
import sys
import os
import argparse
import cv2

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("[CaffeInference]")

def preprocess(imdir,imtype, indim, mean_values, scale_values, input_type):
    if imtype == ".txt":
        im = np.loadtxt(imdir, dtype=np.float32).reshape(tuple(indim))
    elif imtype == ".bin":
        im = np.fromfile(imdir, dtype=np.float32).reshape(tuple(indim))
    elif imtype in ['.jpg', '.png', '.jpeg', '.bmp']:
        im = cv2.imread(imdir, cv2.IMREAD_GRAYSCALE) if input_type == "Gray" else cv2.imread(imdir)
        im = cv2.resize(im, (indim[3], indim[2]), interpolation=cv2.INTER_LINEAR)
        if input_type == "RGB":
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = im.astype(np.float32)
        im = im - mean_values[0] if input_type == "Gray" else im - np.array(mean_values, dtype=np.float32)
        im = im * scale_values[0] if input_type == "Gray" else im * np.array(scale_values, dtype=np.float32)
        im = im if input_type == "Gray" else im.transpose(2, 0, 1)
        im = np.reshape(im, tuple(indim))
    else:
        raise AssertionError("Only support .txt, .bin, .jpg, .png, .jpeg and .bmp data format to net!")
    return im  

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--caffe_root", type=str, default="/workspace/nxu/caffe/python/", help="caffe root")
    parser.add_argument("-p", "--prototxt", type=str, default="/home/nxu/workspace/model/customer/DaHua_DenseNet_169/DenseNet_169.prototxt", help="caffemodel path")
    parser.add_argument("-c", "--caffemodel", type=str, default="/home/nxu/workspace/model/customer/DaHua_DenseNet_169/DenseNet_169.caffemodel", help="caffemodel path")
    parser.add_argument("-o", "--output", type=str, default="./caffe_rslts", help="onnx inference output save path")
    parser.add_argument("-v", "--mean_values", nargs='+', type=float, default=[0., 0., 0.], help="pre-processing mean values")
    parser.add_argument("-s", "--scale_values", nargs='+', type=float, default=[1., 1., 1.], help="pre-processing scale values")
    parser.add_argument("-i", "--input_dir", type=str, default="/home/nxu/workspace/model/customer/xiaoshi/b1d9e136-6c94ea3f.jpg", help="support path, image and tensor, tensor is txt or bin!")
    parser.add_argument("-t", "--input_type", type=str, default='BGR', help="pre-processing network support RGB, BGR and Gray!")
    parser.add_argument("--AllTensor", action='store_true', help="dump all node output tensor!")
    # check arguments
    args = parser.parse_args()
    return args

def main(args):
    #os.system("export LD_LIBRARY_PATH=/home/nxu/workspace/anaconda3/envs/py37/lib:/usr/local/hdf5/lib:$LD_LIBRARY_PATH")
    prototxt = args.prototxt
    dump_path=args.output
    if not os.path.exists(dump_path):
        os.makedirs(dump_path)
    caffemodel = args.caffemodel
    caffe_root = args.caffe_root
    if not os.path.exists(caffe_root):
        raise AssertionError("caffe_root:{} is not exist, please check it!".format(caffe_root))
    sys.path.insert(0, caffe_root)
    try:
        import caffe
    except:
        raise AssertionError("Found no caffe root, please set caffe root in env param!")
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    
    output_lists = net.outputs
    input_names_list = net.inputs
    if args.AllTensor:
        output_lists = set(list(net.blobs.keys()) + output_lists)
    netInputs = {}
    for name in input_names_list:
        blob = net.blobs[name]
        indim = [blob.num, blob.channels, blob.height, blob.width]
        if blob.num > 1:
            raise AssertionError("Not support batch_size > 1 inference, please repair!")
        netInputs[name]=indim
    
    net_inputs_data={}
    if len(input_names_list) > 1:
        image_path = args.input_dir
        if not os.path.isdir(image_path):
            raise AssertionError("Multiple input network input must configure folder!")
        imageLists = os.listdir(image_path)
        for imagename in imageLists:
            imDir = os.path.join(image_path, imagename)
            imnamestrip = os.path.splitext(imagename)[0]
            if imnamestrip in input_names_list:
                indim = netInputs[imnamestrip]
                filetype = os.path.splitext(imagename)[1]
                im = preprocess(imDir,filetype, indim, args.mean_values, args.scale_values, args.input_type)
                net_inputs_data[imnamestrip] = im
            else:
                logger.warning("The name of {} cant not match any network input!".format(imagename))
            if len(net_inputs_data) == len(input_names_list):
                break
    else:
        imDir = args.input_dir
        if os.path.isdir(imDir):
            raise AssertionError("Single input network input must configure file!")
        filetype = os.path.splitext(os.path.basename(imDir))[1]
        im = preprocess(imDir, filetype, netInputs[input_names_list[0]], args.mean_values, args.scale_values, args.input_type)
        net_inputs_data[input_names_list[0]] = im  
    
    if len(net_inputs_data) != len(input_names_list):
         raise ValueError("The number of inputs to the network does not match the number of image to fit!") 
     
    for name in input_names_list:
        net.blobs[name].data[...] = net_inputs_data[name]
    
    net.forward() 
    # for _name, blob in net.blobs.items():
    #     if _name in output_lists:
    #         save_file = os.path.join(dump_path, _name+".txt")
    #         print("Dump: layer:",_name,"---->", save_file)
    #         np.savetxt(save_file, np.array(blob.data, dtype=np.float32).reshape(-1), fmt='%.06f')
    for _name in output_lists:
        save_file = os.path.join(dump_path, _name+".txt")
        print("Dump: layer:",_name,"---->", save_file)
        np.savetxt(save_file, np.array(net.blobs[_name].data, dtype=np.float32).reshape(-1), fmt='%.06f')

if __name__ == "__main__":
    args = parse_args()
    main(args)