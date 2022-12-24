# How to download ImageNet:

- Download [ImageNet 2012](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php) after logging in:

The filenames are:  
Development kit (Task 1 & 2). 2.5MB.
Training images (Task 1 & 2). 138GB. MD5: 1d675b47d978889d74fa0da5fadfb00e
Validation images (all tasks). 6.3GB. MD5: 29b22e2961454d5413ddabcf34fc5622
Test images (all tasks). 13GB. MD5: e1b8681fff3d63731c599df9b4b6fc02 (optional)



- After downloading, extract using the following code with GIT BASH on Windows or Linux terminal on Linux:	
```
# extract train data
mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
# extract validation data
cd ../ && mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
```
WARNING: the above code will extract and DELETE the zip files. So better keep backup in case the extraction gets messed up.

**** if wget doesn't work on windows go to the following link and follow instructions so that wget works with bash on windows:
		
	https://gist.github.com/evanwill/0207876c3243bbb6863e65ec5dc3f058

	In brief the steps are:
		-Download the lastest wget binary for windows from eternallybored (they are available as a zip with documentation, or just an exe)
			link: https://eternallybored.org/misc/wget/
		-If you downloaded the zip, extract all (if windows built in zip utility gives an error, use 7-zip).
		-Rename the file wget64.exe to wget.exe if necessary.
		-Move wget.exe to your Git\mingw64\bin\.




- After extraction and processing val images with the code, we will get train and val folder with class names like this:

/directory/to/imagenet/
                |----train
                      |----n01440764
                      |----...
                |----val
                      |----n01440764
                      |----... 

If not, make sure the wget command ran properly or rerun the wget command.


- Put the train folder, val folder and ILSVRC2012_devkit_t12.tar on "root" directory. For the PyTorch code below, root directory is:

	"./data/imagenet"

- In PyTorch code,

	transform=transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                      transforms.Normalize(mean = [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        traindataset = datasets.ImageNet('./data/imagenet', split='train', transform=transform)
        testdataset = datasets.ImageNet('./data/imagenet', split='val', transform=transform) 