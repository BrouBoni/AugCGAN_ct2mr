## CT-to-MR experiment tips

First you need to prepare data as explained in `datasets` folder.

### Training/test options

Please see `options.py` for training/test flags.

### Training Augmented CycleGAN model

`python train.py --dataroot ../datasets/ct2mr_128/ --name model_128/ --loadSize 128`

AugCGAN is quite memory-intensive so for large images, we recommend training 
with cropped images by using the option `--crop`. This option only works with PNG image.
 
`python train.py --dataroot ../datasets/ct2mr_256/ --name model_256/ --loadSize 256 --crop  `

### Fine-tuning/resume training

To fine-tune a pre-trained model, or resume the previous training, use the 
option `--continue_train` and `--load_pretrain checkpoints/model_256/`

### Testing

`python test.py --chk_path checkpoints/model_256/latest --dataroot ../datasets/ct2mr_256/ --res_dir val_res`

