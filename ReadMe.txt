1.extract_nii_file 					提取每个HBP文件内的nii文件 #- Extracting the .nii files from original dicom folders -#
2.phaseLayerCount.py 				计算各相期层数 #- Counting the slice number of each phase -#
3.cut&save.py					若HBP的原始数据删减了某些层，则mask用这代码来删减那些层 #- Z-axis registration based on HBP phase for other phase images -#
4.extract_reshape_32.py 				肿瘤区域提取并resize到32*32*32 #- Extracting the bounding box containing ROI and reshape it to 32*32*32 -#
5.extract_tumor_accroding_centroid_and_reshape.py 	滑块增强 #- Data augmentation -#
6.excel2txt.py					利用清洗好的数据的excel来制作训练验证的txt #- Data division and recording the sample ID in a txt file -#
7.train.py						训练模型 #- Model training -#
8.plot						画ROCing #- Model evaluation -#
9.grad-cam 3D     神经网络的注意力分配可视化 #- Visualization of attention allocation in neural network -#
