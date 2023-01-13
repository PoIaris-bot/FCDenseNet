# FC-DenseNet
FC-DenseNet for container keyhole segmentation
#### train
```bash
python train.py -e <epochs> -m <model> -b <batch_size>
```
#### test
```bash
python test.py -m <model>
```
#### detect
```bash
python detect.py -m <model> -i <path_to_image> -f <format_of_model>
```
#### parameters
```bash
-m: FCDenseNet56 (default) / FCDenseNet67 / FCDenseNet103
-f: torch (default) / onnx
```