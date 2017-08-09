#python3

from scipy.misc import imsave
import numpy as np

# 解压缩，返回解压后的字典
def unpickle(file):
	import pickle
	fo = open(file, 'rb')
	dict = pickle.load(fo,encoding='bytes')
	fo.close()
	return dict

# 生成训练集图片，如果需要png格式，只需要改图片后缀名即可。

ftra=open('data/train.txt','w')
for j in range(1, 6):
	dataName = "data/data_batch_" + str(j)  # 读取当前目录下的data_batch12345文件，dataName其实也是data_batch文件的路径，本文和脚本文件在同一目录下。
	Xtr = unpickle(dataName)
	print(dataName + " is loading...")

	for i in range(0, 10000):
	img = np.reshape(Xtr[b'data'][i], (3, 32, 32))  # Xtr['data']为图片二进制数据
	img = img.transpose(1, 2, 0)  # 读取image
	picName = 'data/train/' + str(Xtr[b'labels'][i]) + '_' + str(i + (j - 1)*10000) + '.jpg'  # Xtr['labels']为图片的标签，值范围0-9，本文中，train文件夹需要存在，并与脚本文件在同一目录下。
	ftra.write(picName+' '+str(Xtr[b'labels'][i])+'\n')
        imsave(picName, img)
	print(dataName + " loaded.")
ftra.close()
print("test_batch is loading...")

# 生成测试集图片
fval=ftra=open('data/val.txt','w')
testXtr = unpickle("data/test_batch")
for i in range(0, 10000):
	img = np.reshape(testXtr[b'data'][i], (3, 32, 32))
	img = img.transpose(1, 2, 0)
	picName = 'data/test/' + str(testXtr[b'labels'][i]) + '_' + str(i) + '.jpg'
	fval.write(picName+' '+ str(testXtr[b'labels'][i])+'\n')
	imsave(picName, img)
fval.close()
print("test_batch loaded.")
