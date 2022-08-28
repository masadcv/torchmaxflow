
import time
import torch
import numpy as np
import SimpleITK as sitk
from PIL import Image
import matplotlib.pyplot as plt
import torchmaxflow 

def demo_maxflow():
    I = Image.open('data/brain.png')
    Iq = np.asarray(I.convert('L'), np.float32)
    P = np.asarray(Image.open('data/brain_mask.png').convert('L'), np.float32) / 255

    fP = 0.5 + (P - 0.5) * 0.8
    bP = 1.0 - fP
    Prob = np.asarray([bP, fP])
    lamda = 20.0  
    sigma = 10.0
    connectivity = 4

    Iq = torch.from_numpy(Iq).unsqueeze(0).unsqueeze(0)
    Prob = torch.from_numpy(Prob).unsqueeze(0)

    tic = time.time()
    lab = np.squeeze(torchmaxflow.maxflow(Iq, Prob, lamda, sigma, connectivity).numpy())
    toc = time.time()
    print("Time taken: {}".format(toc-tic))
    
    plt.subplot(1,3,1); plt.axis('off'); plt.imshow(I);  plt.title('Input')
    plt.subplot(1,3,2); plt.axis('off'); plt.imshow(fP);   plt.title('Initial Segmentation')
    plt.subplot(1,3,3); plt.axis('off'); plt.imshow(lab); plt.title('Graphcut result')
    plt.show()

def demo_interactive_maxflow():
    I = Image.open('data/brain.png')
    Iq = np.asarray(I.convert('L'), np.float32)
    P = np.asarray(Image.open('data/brain_mask.png').convert('L'), np.float32) / 255

    fP = 0.5 + (P - 0.5) * 0.8
    bP = 1.0 - fP
    Prob = np.asarray([bP, fP])

    S  = np.asarray(Image.open('data/brain_scrb.png').convert('L'))
    Seed = np.asarray([S == 255, S == 170], np.float32)

    lamda = 30.0  
    sigma = 8.0
    connectivity = 4

    Iq = torch.from_numpy(Iq).unsqueeze(0).unsqueeze(0)
    Prob = torch.from_numpy(Prob).unsqueeze(0)
    Seed = torch.from_numpy(Seed).unsqueeze(0)
    tic = time.time()
    lab = np.squeeze(torchmaxflow.maxflow_interactive(Iq, Prob, Seed, lamda, sigma, connectivity).numpy())
    toc = time.time()
    print("Time taken: {}".format(toc-tic))

    # scribbles display - remove foreground
    S[S == 85] = 0  

    plt.subplot(1,4,1); plt.axis('off'); plt.imshow(I);  plt.title('Input')
    plt.subplot(1,4,2); plt.axis('off'); plt.imshow(fP);   plt.title('Initial Segmentation')
    plt.subplot(1,4,3); plt.axis('off'); plt.imshow(S);   plt.title('User-scribbles')
    plt.subplot(1,4,4); plt.axis('off'); plt.imshow(lab); plt.title('Graphcut result')
    plt.show()

def demo_maxflow3d():
    img_name   = "data/2013_12_1_img.nii.gz"
    prob_name  = "data/2013_12_1_init.nii.gz"
    save_name  = "data/seg_auto.nii.gz"
    img_obj  = sitk.ReadImage(img_name)
    img_data = sitk.GetArrayFromImage(img_obj)
    img_data = np.asarray(img_data, np.float32)
    prob_obj = sitk.ReadImage(prob_name)
    prob_data = sitk.GetArrayFromImage(prob_obj)
    prob_data = np.asarray(prob_data, np.float32)

    fP = 0.5 + (prob_data - 0.5) * 0.8
    bP = 1.0 - fP
    Prob = np.asarray([bP, fP])

    lamda = 10.0
    sigma = 15.0
    connectivity = 6

    img_data = torch.from_numpy(img_data).unsqueeze(0).unsqueeze(0)
    Prob = torch.from_numpy(Prob).unsqueeze(0)

    tic = time.time()
    lab = np.squeeze(torchmaxflow.maxflow(img_data, Prob, lamda, sigma, connectivity).numpy())
    toc = time.time()
    print("Time taken: {}".format(toc-tic))

    lab_obj = sitk.GetImageFromArray(lab)
    lab_obj.CopyInformation(img_obj)
    sitk.WriteImage(lab_obj, save_name)
    print('the segmentation has been saved to {0:}'.format(save_name))

def test_interactive_max_flow3d():
    img_name   = "data/2013_12_1_img.nii.gz"
    prob_name  = "data/2013_12_1_init.nii.gz"
    seed_name  = "data/2013_12_1_scrb.nii.gz"
    save_name  = "data/seg_interact.nii.gz"
    img_obj  = sitk.ReadImage(img_name)
    img_data = sitk.GetArrayFromImage(img_obj)
    img_data = np.asarray(img_data, np.float32)
    prob_obj = sitk.ReadImage(prob_name)
    prob_data = sitk.GetArrayFromImage(prob_obj)
    prob_data = np.asarray(prob_data, np.float32)

    fP = 0.5 + (prob_data - 0.5) * 0.8
    bP = 1.0 - fP
    Prob = np.asarray([bP, fP])

    seed_obj  = sitk.ReadImage(seed_name)
    seed_data = sitk.GetArrayFromImage(seed_obj)
    Seed = np.asarray([seed_data == 2, seed_data == 3], np.float32)

    lamda = 10.0
    sigma = 15.0
    connectivity = 6

    img_data = torch.from_numpy(img_data).unsqueeze(0).unsqueeze(0)
    Prob = torch.from_numpy(Prob).unsqueeze(0)
    Seed = torch.from_numpy(Seed).unsqueeze(0)
    tic = time.time()
    lab = np.squeeze(torchmaxflow.maxflow_interactive(img_data, Prob, Seed, lamda, sigma, connectivity).numpy())
    toc = time.time()
    print("Time taken: {}".format(toc-tic))
    lab_obj = sitk.GetImageFromArray(lab)
    lab_obj.CopyInformation(img_obj)
    sitk.WriteImage(lab_obj, save_name)
    print('the segmentation has been saved to {0:}'.format(save_name))

if __name__ == '__main__':
    print("example list")
    print(" 0 -- 2D max flow without interactions")
    print(" 1 -- 2D max flow with interactions")
    print(" 2 -- 3D max flow without interactions")
    print(" 3 -- 3D max flow with interactions")
    print("please enter the index of an example:")
    method = input()
    method = "{0:}".format(method)
    if(method == '0'):
        demo_maxflow()
    elif(method == '1'):
        demo_interactive_maxflow()
    elif(method == '2'):
        demo_maxflow3d()
    elif(method == '3'):
        test_interactive_max_flow3d()
    else:
        print("invalid number : {0:}".format(method))