import os
from bbox_location import write_location, largest_bbox
from extract_tumor_accroding_centroid_and_reshape import read_dicom_series,extract_tumor_bbox
from tqdm import tqdm


def main():
    '''
    作用是将数据中的肿瘤区域提取并reshape成32*32*32的大小

    :return:
    '''
    location_txt_name = 'bbox_location'
    save_extract_path=r'H:\Bayer\data3\extract_roi_shape'
    save_val_path=r'H:\Bayer\data3\val'
    mask_path = r'E:\LSX\Bayer\SYS1_PV\CK19_nii'
    tumor_data_path = r'E:\LSX\Bayer\SYS1_PV\CK19'

    if os.path.exists(save_extract_path)==False:
        os.makedirs(save_extract_path)
    if os.path.exists(save_val_path)==False:
        os.makedirs(save_val_path)
    tumor_id = os.listdir(mask_path)

    for i in tumor_id:
        roi_path = os.path.join(mask_path, i)
        if os.path.exists(location_txt_name)==False:
            '''
            根据勾画标签 '.nii'文件来制作bbox_location.txt
            当已存在时，则不写入
            可以改成每次都重新制作，这样方便增加数据
            '''
            write_location(roi_path, location_txt_name, i)

    patient_id = os.listdir(tumor_data_path)

    for i in patient_id:
        data_path_1 = os.path.join(tumor_data_path, i)
        data_path_2 = os.listdir(data_path_1)
        for j in data_path_2:
            data_path_3 = os.path.join(data_path_1, j)

            data_path_4=os.listdir(data_path_3)     #新数据的存储路径不一致
            if len(data_path_4)==1:
                data_path_3=os.path.join(data_path_3,data_path_4[0])

            values = read_dicom_series(data_path_3) #按序列读取dicom文件
            j_2=j.split(' ')
            save_extract_path_2=os.path.join(save_extract_path,j_2[-1])
            save_val_path_2=os.path.join(save_val_path,j_2[-1])
            if os.path.exists(save_extract_path_2) == False:
                os.makedirs(save_extract_path_2)
            if os.path.exists(save_val_path_2) == False:
                os.makedirs(save_val_path_2)
            extract_tumor_bbox(location_txt_name,values,save_extract_path_2, save_val_path_2,i)  #提取肿瘤区域并保存

if __name__=="__main__":
    main()