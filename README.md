# circuit-analysis
학습 과정에서 모델 내부의 circuit을 분석하기 위함.

- We use [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) library

## 1. Setup
### conda
- This will install latest python version
```bash
conda create -n circuit-analysis python
conda activate circuit-analysis
```
### poetry
#### Install
```bash
sudo curl -sSL https://install.python-poetry.org | python3 -
```
#### Setup Env. Variable
- Add installed path in `.bashrc`(or `.zshrc` something else)
- For me, it was `~/.local/bin`
```bash
export PATH="[home directory path]/.local/bin:$PATH"
```
```bash
source .bashrc
```
#### Install TransformerLens module in virtual environment
```bash
conda activate circuit-analysis
cd [the path of circuit-analysis/Transformerlens directory]
poetry install
```


