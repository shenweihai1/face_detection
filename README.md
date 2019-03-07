### face_detection
using the pre-trained model from [face_alignment](https://github.com/1adrianb/face-alignment) to recognize the face object from the input image based on [How far are we from solving the 2D \& 3D Face Alignment problem](https://www.adrianbulat.com/face-alignment). **using the others' model, not written by myself!**
![IMAGE ALT TEXT HERE](https://raw.githubusercontent.com/shenweihai1/face_detection/master/assets/face_dection.jpg)


### installation
```
conda install -c 1adrianb face_alignment --name=conda_envs
conda install PIL --name=conda_envs
```

### data-set
LS3D-W is a large-scale 3D face alignment dataset constructed by annotating the images from AFLW[2], 300VW[3], 300W[4] and FDDB[5] in a consistent manner with 68 points using the automatic method described in [1]. To gain access to the dataset please enter your email address in the following form. You will shortly receive an email at the specified address containing the download link
``` 
References: 
[1] A. Bulat, G. Tzimiropoulos. How far are we from solving the 2D & 3D Face Alignment problem? (and a dataset of 230,000 3D facial landmarks), arxiv, 2017 
[2] M. Kostinger, P. Wohlhart, P.M. Roth, and H. Bischof. Annotated facial landmarks in the wild: A large-scale, real-world database for facial landmark localization, In ICCVW, 2011 
[3] J. Shen, S. Zafeiriou, G. G. Chrysos, J. Kossaifi, G. Tzimiropoulos, and M. Pantic. The first facial landmark tracking in-the-wild challenge: Benchmark and results. In ICCVW, 2015 
[4] C. Sagonas, G. Tzimiropoulos, S. Zafeiriou, and M. Pantic. 300 faces in-the-wild challenge: The first facial landmark localization challenge. In ICCVW, 2013 
[5] V. Jain, E. Learned-Miller FDDB: A Benchmark for Face Detection in Unconstrained Settings. UMass Amherst Technical Report, 2010
```

### tips
When try to function `get_landmarks_from_image` to predict:
>
> AttributeError: module 'torch.nn.functional' has no attribute 'interpolate'

refer to [[commit](https://github.com/Skorkmaz88/face-alignment/commit/7ca0f9eaa8020110d529b81a6853a6c01e672472)]
