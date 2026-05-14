import net
import torch
import os
from face_alignment import align
import numpy as np


adaface_models = {
    'ir_50':"pretrained/adaface_ir50_ms1mv2.ckpt",
}

def load_pretrained_model(architecture='ir_50'):
    # load model and pretrained statedict
    assert architecture in adaface_models.keys()
    model = net.build_model(architecture)
    statedict = torch.load(adaface_models[architecture])['state_dict']
    model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    return model

def to_input(pil_rgb_image):
    np_img = np.array(pil_rgb_image)
    print(f"{np_img.shape=}, {np_img.dtype=}, {np_img.max()=}, {np_img.min()=}") 
    # np_img.shape=(112, 112, 3), np_img.dtype=dtype('uint8'), np_img.max()=253, np_img.min()=2
    brg_img = ((np_img[:,:,::-1] / 255.) - 0.5) / 0.5
    print(f"{brg_img.shape=}, {brg_img.dtype=}, {brg_img.max()=}, {brg_img.min()=}")
    # brg_img.shape=(112, 112, 3), dtype=dtype('float64'), max=1, min=-1
    tensor = torch.tensor([brg_img.transpose(2,0,1)]).float()
    print(f"{tensor.size()=}, {tensor.dtype=}, {tensor.max()=}, {tensor.min()=}")
    # 1,3,112,112 torch.float32 max=1, min=-1
    return tensor

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'

    model = load_pretrained_model('ir_50')
    feature, norm = model(torch.randn(2,3,112,112))

    input_images = {
        'LQ': '/data/user/jkwang/DFOSD/data/CelebA-Test/LQ/self_celeba_512_v2/00000000.png', 
        'HQ': '/data/user/jkwang/DFOSD/data/CelebA-Test/HQ/celeba_512_validation/00000000.png',
        'CodeFormer': '/data/user/jkwang/DFOSD/results_zhanglin/CodeFormer/celebA_LQ/restored_faces/00000000.png',
        'DiffBIR': '/data/user/jkwang/DFOSD/results_zhanglin/DiffBIR/celebA_LQ/00000000.png', 
        'RestoreFormer++': '/data/user/jkwang/DFOSD/results_zhanglin/RestoreFormer++/celebA_LQ/aligned/restored_faces/00000000_00.png', 
        'PGDiff': '/data/user/jkwang/DFOSD/results_zhanglin/PGDiff/celebTest_A/s0.05-seed1234/00000000.png', 
        'DAEFR': '/data/user/jkwang/DFOSD/results_zhanglin/DAEFR/self_celeba_512_v2/restored_faces/00000000_00.png', 
    }

    test_image_path = 'face_alignment/test_images'
    features = []
    # for fname in sorted(os.listdir(test_image_path)):
        # path = os.path.join(test_image_path, fname)
    for bname, path in input_images.items():
        print(f"{bname=}, {path=}")
        aligned_rgb_img = align.get_aligned_face(path)
        bgr_tensor_input = to_input(aligned_rgb_img)
        feature, _ = model(bgr_tensor_input)
        features.append(feature)
        print(f"{bname} feature shape: {feature.shape}")
        print(f"{bname} features shape: {torch.cat(features).shape}")

    similarity_scores = torch.cat(features) @ torch.cat(features).T
    print(similarity_scores)

    aligned_rgb_img = align.get_aligned_face(input_images['LQ'])
    bgr_tensor_input = to_input(aligned_rgb_img)
    feature0, _ = model(bgr_tensor_input)

    aligned_rgb_img = align.get_aligned_face(input_images['HQ'])
    print(f"{type(aligned_rgb_img)=}")
    print(f"{aligned_rgb_img.size=}, max: {aligned_rgb_img.getextrema()}")
    bgr_tensor_input = to_input(aligned_rgb_img)
    print(f"{bgr_tensor_input.size()=}, max: {bgr_tensor_input.max()}, min: {bgr_tensor_input.min()}")
    print(f"{type(bgr_tensor_input)=}")
    feature1, _ = model(bgr_tensor_input)

    similarity_score = feature0 @ feature1.T
    print(similarity_score)
    

