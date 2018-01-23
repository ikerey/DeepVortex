# DeepVortex

`DeepVortex` is an end-to-end deep neural network to detect vortex flows from pairs of images containing the x and y component of the horizontal velocities. An update of the neural network model will be provided soon.

![example](Figs/vortex_detection_DeepVortex.gif?raw=true "")
**Figure 1** - Example of vortex flow detection with `DeepVortex`.

## Dependencies of `DeepVortex`

`DeepVortex` requires some non-standard libraries: Keras (v2) and TensorFlow.
If `conda` is installed run:

* `conda install -c anaconda tensorflow` (for CPU)
* `conda install -c anaconda tensorflow-gpu` (for GPU)
* `conda install keras`
 
## Using `DeepVortex` for prediction

We provide a pre-trained model of `DeepVortex` (an updated model will be provided soon). You can use it from the command line by typing: 

```
python deepvortex.py -i sample/sample.fits -o output/output.fits
```
  
We provide the sample file containing 100 frames from the MURaM simulations. The file is a FITS file containing an array of size `(n_frames x nx x ny x 2)`. The first dimension is the number of frames. The second and third dimensions are the size of the input image. Finally, the last dimension contains the vx and vy velocity maps. The velocity maps must be normalized to the range [-1, 1].

## Using `DeepVortex` for training

If you want to train `DeepVortex` with your own images, we provide the script `train_deepvortex.py` to this aim.

```
python train_deepvortex.py -a start -e 20 -o network/model -n 1e-3
``` 

The parameters are:

    -a={start,continue}
        `start`: start a new calculation
        `continue`: continue a previous calculation
    -e=20
        Number of epochs to use during training
    -o=network/model 
        Define the output file that will contain the network topology and weights
    -n=1e-3
        Noise to add during training

We do not provide training data in this repository.
