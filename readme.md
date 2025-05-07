### generation stage

#### Requirements
```
pip install -r requirements.txt

pip install ./simple-knn
pip install git+https://github.com/NVlabs/nvdiffrast/
pip install git+https://github.com/ashawkey/kiuikit

git clone https://github.com/openai/shap-e.git
cd shap-e
pip install -e .

#### Usage
python main.py --config configs/text.yaml prompt="a airplane" save_path=airplane

```
The generated data is placed in the xxx/RISurConv-aux/data path for classification

### classification stage
#### Requirements
'''
This repo provides source codes, which had been tested with PyTorch 1.10.0, CUDA 11.3 on Ubuntu 20.04.

cd pointops
python3 setup.py install
cd ..
'''
#### Usage
python train_classification.py 