# caffe_convert_onnx

**We have developed a set of tools for converting caffemodel to onnx model to facilitate the deployment of algorithms on mobile platforms.

**However, due to the company secrets involved, we can only provide compiled executable files.

**And provide a script to perform forward reasoning on the onnx model and the original caffe model to compare whether the converted results are normal.

### Note: Our engineering support is not limited to the conversion of the following caffe operators

| Input                | LRN          | Deconvolution  |
| -------------------- | ------------ | -------------- |
| Input                | InnerProduct | Interp         |
| VideoData            | Reshape      | Split          |
| Convolution          | Transpose    | Slice          |
| Convolution3D        | Gemm         | ShuffleChannel |
| ConvolutionDepthwise | DropOut      | Axpy           |
| DepthwiseConvolution | Concat       | Crop           |
| BatchNorm            | Swish        | Power          |
| Normalize            | Sigmoid      | Im2Col         |
| BN                   | Mish         | Transpose      |
| Scale                | BroadcastMul | Permute        |
| ReLU                 | Eltwise      | Lstm           |
| PReLU                | Flatten      | LSTM           |
| Pooling              | MaxUnpool    | Reverse        |
| Pooling3D            | Upsample     | Reorg          |
| Softmax              | Uppooling    | SpatialProduct |
| AbsVal               | Threshold    | Reduction      |

## **How to use our project?**

Congratulations, if you simply convert the caffe model to the onnx model, you don't need to configure any environment. Just run the executable file we provide under the ubuntu system.

```bash
git clone https://github.com/xncaffe/caffe_convert_onnx.git
cd caffe_convert_onnx/cmd
./convert_main --prototxt ../examples/inference/models/caffe/mobilenet_v1/deploy.prototxt \
		--caffemodel ../examples/inference/models/caffe/mobilenet_v1/deploy.caffemodel \
		--out ../examples/inference/models/onnx/mobilenet_v1.onnx
```

If you are in China, the clone project is slow and often interrupted, you can consider using the download mirror gitclone.com, refer to the command:

```bash
git clone https://gitclone.com/github.com/xncaffe/caffe_convert_onnx.git
```

Or get the conversion executable tool convert_main from the URL [https://download.csdn.net/download/xunan003/87659000?spm=1001.2014.3001.5503](https://download.csdn.net/download/xunan003/87659000?spm=1001.2014.3001.5503 "https://download.csdn.net/download/xunan003/87659000?spm=1001.2014.3001.5503")

You can also use -h for help

```bash
./convert_main -h
Found no caffe root, please set caffe root in env param! If transplanting an onnx model, please ignore this warning.
usage: convert_main [-h] -p PROTOTXT [-c CAFFEMODEL] [-o OUT]

optional arguments:
  -h, --help            show this help message and exit
  -p PROTOTXT, --prototxt PROTOTXT
                        deploy.prototxt path
  -c CAFFEMODEL, --caffemodel CAFFEMODEL
                        deploy.caffemodel path
  -o OUT, --out OUT     onnx model output path

```

### Note Other

If you need to use the forward reasoning program we provide to infer the caffe model and onnx model?

**You need to configure some environments to support their normal operation.

1. First you need to install a working caffe framework and configure its python interface.
2. The python version >= 3.7, we used python 3.7.13.
3. Download the example caffemodel from address [https://download.csdn.net/download/xunan003/87658946?spm=1001.2014.3001.5503](https://download.csdn.net/download/xunan003/87658946?spm=1001.2014.3001.5503 "https://download.csdn.net/download/xunan003/87658946?spm=1001.2014.3001.5503"), And decompress examples.zip and place it in the *caffe_convert_onnx* project directory.
4. Install dependencies according to the provided requirements.txt.

   ```bash
   pip install -r requirements.txt
   ```
5. Follow the instructions below to get help on using the inference caffe and onnx models

   ```
   python caffe_inference.py -h

   usage: caffe_inference.py [-h] [-r CAFFE_ROOT] [-p PROTOTXT] [-c CAFFEMODEL]
                             [-o OUTPUT] [-v MEAN_VALUES [MEAN_VALUES ...]]
                             [-s SCALE_VALUES [SCALE_VALUES ...]] [-i INPUT_DIR]
                             [-t INPUT_TYPE] [--AllTensor]

   optional arguments:
     -h, --help            show this help message and exit
     -r CAFFE_ROOT, --caffe_root CAFFE_ROOT
                           caffe root
     -p PROTOTXT, --prototxt PROTOTXT
                           caffemodel path
     -c CAFFEMODEL, --caffemodel CAFFEMODEL
                           caffemodel path
     -o OUTPUT, --output OUTPUT
                           onnx inference output save path
     -v MEAN_VALUES [MEAN_VALUES ...], --mean_values MEAN_VALUES [MEAN_VALUES ...]
                           pre-processing mean values
     -s SCALE_VALUES [SCALE_VALUES ...], --scale_values SCALE_VALUES [SCALE_VALUES ...]
                           pre-processing scale values
     -i INPUT_DIR, --input_dir INPUT_DIR
                           support path, image and tensor, tensor is txt or bin!
     -t INPUT_TYPE, --input_type INPUT_TYPE
                           pre-processing network support RGB, BGR and Gray!
     --AllTensor           dump all node output tensor!

   ```

   ```
   python onnx_inference.py -h
   usage: onnx_inference.py [-h] [-m ONNXMODEL] [-o OUTPUT]
                            [-v MEAN_VALUES [MEAN_VALUES ...]]
                            [-s SCALE_VALUES [SCALE_VALUES ...]] [-i INPUT_DIR]
                            [-t INPUT_TYPE] [--AllTensor]

   optional arguments:
     -h, --help            show this help message and exit
     -m ONNXMODEL, --onnxmodel ONNXMODEL
                           onnx model path
     -o OUTPUT, --output OUTPUT
                           onnx inference output save path
     -v MEAN_VALUES [MEAN_VALUES ...], --mean_values MEAN_VALUES [MEAN_VALUES ...]
                           pre-processing mean values
     -s SCALE_VALUES [SCALE_VALUES ...], --scale_values SCALE_VALUES [SCALE_VALUES ...]
                           pre-processing scale values
     -i INPUT_DIR, --input_dir INPUT_DIR
                           support path, image and tensor, tensor is txt or bin!
     -t INPUT_TYPE, --input_type INPUT_TYPE
                           pre-processing network support RGB, BGR and Gray!
     --AllTensor           dump all node output tensor!

   ```
6. We explain the parameters of caffe and onnx reasoning as follows.

   #### *caffe*

   **caffe_root** -> This is what you have to configure. You need to install the caffe package and compile its python interface, and then configure it. For example, --caffe_root=/home/caffe/python

   **mean_values** -> This is the mean parameter necessary for pre-processing, the default is [0, 0, 0], which strictly corresponds to the format of your --input_type configuration. If --input_type=RGB, the corresponding relationship is r_mean_value=mean_values[0], g_mean_value=mean_values[1], b_mean_value=mean_values[2]. If --input_type=BGR then b_mean_value=mean_values[0], r_mean_value=mean_values[2]. If --input_type=Gray, y_mean_value=mean_values[0].

   **scale_values** -> Similar to mean_values, it is a parameter for pre-processing normalization, and the default value is [1, 1, 1]. Same as mean_values, its order strictly corresponds to --input_type.

   **input_dir** -> Network input file, which can be a path or a specific file. The single-input network is a file, and the multi-input network must be a folder.

   **input_type** -> As explained in mean_values, support RGB\BGR and Gray input, configure according to the actual input of the network. Default is RGB.

   **AllTensor** -> Turn it on if you want to spit out the output of all layers.

   #### ONNX

   Except that caffe_root does not need to be configured, other parameters are consistent with caffe.

   **Note:** Our pre-processing formula is y=(x-mean_values)*scale_values.
7. Example of forward inference using two modelsexamples.

   ```
   python caffe_inference.py --caffe_root ../caffe/python/ \
   			--onnxmodel ./examples/inference/models/onnx/twoInputNetDemo.onnx \
   			--output ./dump_rslts/ \
   			--mean_values 127.5 127.5 127.5 \
   			--scale_values 0.0078 0.0078 0.0078 \
   			--input_dir ./data/multi_input/image$ \
   			--input_type BGR
   ### Caffe multi input network image input ###
   ```

   ```
   python caffe_inference.py --caffe_root ../caffe/python/ \
   			--onnxmodel ./examples/inference/models/onnx/mobilenet_v1.onnx \
   			--output ./dump_rslts/ \
   			--input_dir ./data/1x3x224x224_float32.txt \
   ### Caffe single input network txt input ###
   ```

   ```
   python caffe_inference.py --caffe_root ../caffe/python/ \
   			--onnxmodel ./examples/inference/models/onnx/mobilenet_v1.onnx \
   			--output ./dump_rslts/ \
   			--input_dir ./data/1x3x224x224_float32.bin \
   ### Caffe single input network bin input ###
   ```

   ```
   python caffe_inference.py --caffe_root ../caffe/python/ \
   			--onnxmodel ./examples/inference/models/onnx/mobilenet_v1.onnx \
   			--output ./dump_rslts/ \
   			--mean_values 127.5 127.5 127.5 \
   			--scale_values 0.0078 0.0078 0.0078 \
   			--input_dir ./data/1.jpg \
   			--input_type BGR
   ### Caffe single input network image input ###
   ```

   ```python
   python onnx_inference.py --onnxmodel ./examples/inference/models/onnx/mobilenet_v1.onnx \
   			--output ./dump_rslts/ \
   			--mean_values 127.5 127.5 127.5 \
   			--scale_values 0.0078 0.0078 0.0078 \
   			--input_dir ./data/1.jpg \
   			--input_type BGR
   ### Onnx single input network image input ###
   ```

   For other tests, please operate similarly.

Note that multi-input networks only support the case where all inputs are the same set of pre-processing parameters.
