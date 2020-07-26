# speech-enhancement-with-cnns
Speech Enhancement With CNNs: PyTorch code with Pretrained model. 

1. Clone the repo. `git clone https://github.com/abhishekyana/speech-enhancement-with-cnns` and cd into it. 
1. Install all the requirements mentioned in the `requirements.txt`
1. CD into INFERRING - `cd ./INFERRING`
1. Run `python infer.py -in <input_audio> -model ./finalmodel.mdl -cuda` # make sure Torch with Cuda is installed. 
1. Denoised audio file will be saved in `./AudioOuts` as `clean_{filename}`.


Thank you
