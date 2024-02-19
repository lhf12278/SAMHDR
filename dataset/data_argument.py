from utils import *
import imgvision as iv
from PIL import Image

# scene_directory = r'F:\yzm\data1\Test\EXTRA'
scene_directory = r'F:\yzm\data\train_s'

def xyz2ipt(xyz_image):
    """convert xyz into IPT"""
    if len(xyz_image.shape) != 3 or xyz_image.shape[2] != 3:
        raise ValueError("input image is not a xyz image")
    xyz_image = xyz_image.astype(np.float32)
    # 1：创建变换矩阵，和偏移量
    transform_matrix = np.array([[0.4002, 0.7075, -0.0807],
                                 [-0.2280, 1.1500, 0.0612],
                                 [0.0000, 0.0000, 0.9184]])
    transform_matrix1 = np.array([[0.4000, 0.4000, 0.2000],
                                 [4.4550, -4.8510, 0.3960],
                                 [0.8056, 0.3572, -1.1628]])
    LMS_image = np.zeros(shape=xyz_image.shape)
    LMS_image1 = np.zeros(shape=xyz_image.shape)
    w, h, c = xyz_image.shape
    for i in range(w):
        for j in range(h):
            LMS_image[i, j, :] = np.dot(transform_matrix, xyz_image[i, j, :])
            for k in range(c):
                if LMS_image[i, j, k] >= 0:
                    LMS_image1[i, j, k] = LMS_image[i, j, k] ** 0.43
                else:
                    LMS_image1[i, j, k] = - (abs(LMS_image[i, j, k]) ** 0.43)
    IPT_image = np.zeros(shape=xyz_image.shape)
    for i in range(w):
        for j in range(h):
            IPT_image[i, j, :] = np.dot(transform_matrix1, LMS_image1[i, j, :])
    # I = IPT_image[:, :, 0]
    P = IPT_image[:, :, 1]
    T = IPT_image[:, :, 2]
    PT = np.sqrt(np.add(P**2, T**2))
    Ms = np.zeros(shape=PT.shape)
    W, H = PT.shape
    for i in range(W):
        for j in range(H):
            if PT[i, j] >= 1:
                Ms[i, j] = 1
            else:
                Ms[i, j] = PT[i, j]
    return Ms

if __name__ == '__main__':

    list = os.listdir(scene_directory)
    num = 0
    for scene in range(len(list)):
        file_path = list_all_files_sorted(os.path.join(scene_directory, list[scene]), '.tif')
        imgs = ReadImages(file_path)
        img = imgs[1]
        cvtor = iv.cvtcolor()
        # name = file_path[2][-12:-4]
        # save_path = file_path[1][0:-12]
        name = file_path[1][-19:-4]
        save_path = file_path[1][0:-19]
        xyz_Image = cvtor.rgb2xyz(img)  # RGB转XYZ
        Ms = xyz2ipt(xyz_Image)  # XYZ转IPT并生成饱和压缩图
        Ms = torch.from_numpy(Ms)  # 数组转tensor
        Ms = Ms.unsqueeze(dim=0).permute(1, 2, 0).numpy()  # 增加通道维度并转为数组
        Ms = (Ms * 255).astype(np.uint8)
        cv2.imwrite(save_path + name + '.jpg', Ms)
        num = num + 1
