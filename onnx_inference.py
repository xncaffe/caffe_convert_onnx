import onnx
import os
import sys
import cv2

import argparse
import numpy as np
import json
import onnxruntime as rt
import copy
from collections import OrderedDict
from onnx import TensorProto
import matplotlib.pyplot as plt
from onnx import numpy_helper as nph
from onnxsim import simplify

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("[ONNXReference]")

NPDTYPE_2_ONNXDTYPE = {
    'float32': TensorProto.FLOAT,   #index = 1
    'uint8': TensorProto.UINT8,     #index = 2
    'int8': TensorProto.INT8,       #index = 3
    'uint16': TensorProto.UINT16,   #index = 4
    'int16': TensorProto.INT16,     #index = 5
    'int32': TensorProto.INT32,     #index = 6
    'int64': TensorProto.INT64,     #index = 7
    'object': TensorProto.STRING,   #index = 8
    '<U0': TensorProto.STRING,      #index = 8
    'bool': TensorProto.BOOL,       #index = 9
    'float16': TensorProto.FLOAT16, #index = 10
    'float64': TensorProto.DOUBLE,  #index = 11
    'uint32': TensorProto.UINT32,   #index = 12
    'uint64': TensorProto.UINT64,   #index = 13
    np.dtype(np.float32): TensorProto.FLOAT,    #index = 1
    np.dtype(np.uint8): TensorProto.UINT8,      #index = 2
    np.dtype(np.int8): TensorProto.INT8,        #index = 3
    np.dtype(np.uint16): TensorProto.UINT16,    #index = 4
    np.dtype(np.ushort): TensorProto.UINT16,    #index = 4
    np.dtype(np.short): TensorProto.INT16,      #index = 5
    np.dtype(np.int16): TensorProto.INT16,      #index = 5
    np.dtype(np.int32): TensorProto.INT32,      #index = 6
    np.dtype(np.int64): TensorProto.INT64,      #index = 7
    #np.dtype(np.int): TensorProto.INT64,        #index = 7
    np.dtype(np.str_): TensorProto.STRING,       #index = 8
    #np.dtype(np.bool): TensorProto.BOOL,        #index = 9
    np.dtype(np.bool_): TensorProto.BOOL,       #index = 9
    np.dtype(np.bool8): TensorProto.BOOL,       #index = 9
    np.dtype(np.float16): TensorProto.FLOAT16,  #index = 10
    #np.dtype(np.float): TensorProto.DOUBLE,     #index = 11
    np.dtype(np.float64): TensorProto.DOUBLE,   #index = 11
    np.dtype(np.uint32): TensorProto.UINT32,    #index = 12
    np.dtype(np.uint64): TensorProto.UINT64,    #index = 13
    np.dtype(np.uint): TensorProto.UINT64,      #index = 13
}

def forward_by_onnxruntime(onnx_model):
    ort_session = rt.InferenceSession(onnx_model.SerializeToString())
    inputs = ort_session.get_inputs()
    ort_inputs={}
    for net_input_index in range(len(ort_session.get_inputs())):
        net_input=onnx_model.graph.input[net_input_index]
        input_shape = net_input.type.tensor_type.shape.dim
        input_shape_list=[]
        for _i in input_shape:
            input_shape_list.append(_i.dim_value)

        if(input_shape_list[0]==0):
            input_shape_list[0]=1
        img_array = np.zeros(tuple(input_shape_list), dtype = np.float32)
        ort_inputs[ort_session.get_inputs()[net_input_index].name]=img_array
    outputs = [x.name for x in ort_session.get_outputs()]
    ort_outs = ort_session.run(outputs, ort_inputs)

    return OrderedDict(zip(outputs, ort_outs)) 
    

def infer_model_shape(onnx_model):
    onnx_model_all_output=copy.deepcopy(onnx_model)
    onnx_model_shape=copy.deepcopy(onnx_model)
    del onnx_model_shape.graph.value_info[:]
    del onnx_model_shape.graph.output[:]
    onnx.save(onnx_model_shape, "del_allnode.onnx")
    ori_model_output_list=[]
    for out in onnx_model.graph.output:
        ori_model_output_list.append(out.name)
    for node in onnx_model_all_output.graph.node:
        for output in node.output:
            if output not in ori_model_output_list:
                onnx_model_all_output.graph.output.extend([onnx.ValueInfoProto(name=output)])
    ort_outs = forward_by_onnxruntime(onnx_model_all_output)
    #onnx.save(onnx_model_all_output, "add_alloutput.onnx")
    
    for node in onnx_model_shape.graph.node:
        for output in node.output:
            use_value_info = onnx.helper.make_tensor_value_info(output, NPDTYPE_2_ONNXDTYPE[str(ort_outs[output].dtype)], ort_outs[output].shape)
            onnx_model_shape.graph.value_info.append(use_value_info)
    #onnx.save(onnx_model_shape, "add_alloutput_model_shape.onnx")
    return onnx_model_shape

def model_add_outputs(onnx_model):  
    outputs=[]
    ori_outputs=[]
    onnx_shape_model=infer_model_shape(onnx_model)

    for out in onnx_model.graph.output:
        ori_outputs.append(out.name)

    for node_id,node in enumerate(onnx_model.graph.node):
        current_output=node.output
        for out in current_output:
            if out not in ori_outputs:
                outputs.append(out)

    for output in outputs:
        output_shape=get_shape_by_name(onnx_shape_model, output)
        shape=["batch_size"]
        shape.extend(output_shape[1:])
        # output_tensor_new = onnx.helper.make_tensor_value_info(name = output, elem_type = 1, \
        #                                                 shape = shape)
        output_tensor_new = onnx.ValueInfoProto(name = output)
        onnx_model.graph.output.insert(0, output_tensor_new) 

    return onnx_model

def get_shape_by_name(onnx_model,name):
    # search
    graph_input=onnx_model.graph.input
    for value in graph_input:
        if value.name==name:
            dim_list=[]
            for dim in value.type.tensor_type.shape.dim:
                dim_list.append(int(dim.dim_value))
            if(dim_list[0]==0):
                dim_list[0]=1
            return dim_list

    value_info=onnx_model.graph.value_info
    for value in value_info:
        if value.name==name:
            dim_list=[]
            for dim in value.type.tensor_type.shape.dim:
                dim_list.append(int(dim.dim_value))
            if(len(dim_list)>0 and dim_list[0]==0):
                dim_list[0]=1
            return dim_list

    graph_output=onnx_model.graph.output
    for value in graph_output:
        if value.name==name:
            dim_list=[]
            for dim in value.type.tensor_type.shape.dim:
                dim_list.append(int(dim.dim_value))
            if(dim_list[0]==0):
                dim_list[0]=1
            return dim_list

    tensor=get_tensor_from_initializer(onnx_model,name)
    dim_list=[]
    try:
        for s in tensor.shape:
            dim_list.append(int(s))
        return dim_list
    except:
        return [1]

def get_tensor_from_initializer(onnx_model,name):
    for init in onnx_model.graph.initializer:
        if init.name == name:
            return onnx.numpy_helper.to_array(init)
    
    for node in onnx_model.graph.node:
        if node.op_type=="Constant" and name==node.output[0]:
            return onnx.numpy_helper.to_array(node.attribute[0].t)
    
    return []

def get_node_by_output(onnx_model,output_list):
    nodes=[]
    for i in output_list:
        for node in onnx_model.graph.node:
            if i in node.output:
                nodes.append(node)
    return nodes

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
        im = im if input_type == "Gray" else im.transpose(2, 1, 0)
        im = np.reshape(im, tuple(indim))
    else:
        raise AssertionError("Only support .txt, .bin, .jpg, .png, .jpeg and .bmp data format to net!")
    return im        

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--onnxmodel", type=str, default="/workspace/nxu/project/caffe_convert_onnx/caffe_convert_onnx/examples/inference/models/onnx/twoInputNetDemo.onnx", help="onnx model path")
    parser.add_argument("-o", "--output", type=str, default="/workspace/nxu/project/caffe_convert_onnx/caffe_convert_onnx/dump_rslts/onnx_rslts", help="onnx inference output save path")
    parser.add_argument("-v", "--mean_values", nargs='+', type=float, default=[0., 0., 0.], help="pre-processing mean values")
    parser.add_argument("-s", "--scale_values", nargs='+', type=float, default=[1., 1., 1.], help="pre-processing scale values")
    parser.add_argument("-i", "--input_dir", type=str, default="/workspace/nxu/project/caffe_convert_onnx/caffe_convert_onnx/data/multi_input/bin", help="support path, image and tensor, tensor is txt or bin!")
    parser.add_argument("-t", "--input_type", type=str, default='RGB', help="pre-processing network support RGB, BGR and Gray!")
    parser.add_argument("--AllTensor", action='store_true', help="dump all node output tensor!")
    # check arguments
    args = parser.parse_args()
    return args

def main(args):
    onnx_path=args.onnxmodel
    dump_path=args.output
    if not os.path.exists(dump_path):
        os.makedirs(dump_path)
    
    onnx_model=onnx.load(onnx_path)
    netInputs = {}
    inputnum = len(onnx_model.graph.input)
    for i in range(inputnum):
        input = onnx_model.graph.input[i]
        netInputs[input.name] = []
        dimNum = len(input.type.tensor_type.shape.dim)
        dims = input.type.tensor_type.shape.dim
        for j in range(dimNum):
            dim = dims[j].dim_value if dims[j].dim_value != 0 else dims[j].dim_param
            dim = 1 if dim == "batch_size" else dim
            netInputs[input.name].append(dim)
    if args.AllTensor:
        onnx_model=model_add_outputs(onnx_model)
    ort_inputs={}
    ort_session = rt.InferenceSession(onnx_model.SerializeToString())
    if inputnum > 1:
        image_path = args.input_dir
        if not os.path.isdir(image_path):
            raise AssertionError("Multiple input network input must configure folder!")
        imageLists = os.listdir(image_path)
        for imagename in imageLists:
            imDir = os.path.join(image_path, imagename)
            imnamestrip = os.path.splitext(imagename)[0]
            if imnamestrip in netInputs:
                indim = netInputs[imnamestrip]
                filetype = os.path.splitext(imagename)[1]
                im = preprocess(imDir,filetype, indim, args.mean_values, args.scale_values, args.input_type)
                ort_inputs[imnamestrip] = im
            else:
                logger.warning("The name of {} cant not match any network input!".format(imagename))
            if len(ort_inputs) == inputnum:
                break
    else:
        imDir = args.input_dir
        if os.path.isdir(imDir):
            raise AssertionError("Single input network input must configure file!")
        filetype = os.path.splitext(os.path.basename(imDir))[1]
        im = preprocess(imDir, filetype, netInputs[ort_session.get_inputs()[0].name], args.mean_values, args.scale_values, args.input_type)
        ort_inputs[ort_session.get_inputs()[0].name] = im  
    if len(ort_inputs) != inputnum:
         raise ValueError("The number of inputs to the network does not match the number of image to fit!") 
    outputs = [x.name for x in ort_session.get_outputs()]
    ort_outs = ort_session.run(outputs, ort_inputs)
    out_dict=OrderedDict(zip(outputs, ort_outs))
    for key in out_dict.keys():
        out_array=out_dict[key]
        node_name=get_node_by_output(onnx_model,[key])[0].name
        out_path=os.path.join(dump_path,str(node_name+"_"+key+".txt").replace("::", "_"))
        print("Dump: layer:",key,"---->",out_path)
        np.savetxt(out_path,out_array.reshape(-1), fmt='%.06f')


if __name__ == "__main__":
    args = parse_args()
    main(args)