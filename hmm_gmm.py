from python_speech_features import *
from scipy.io import wavfile
from hmmlearn import hmm
from sklearn.externals import joblib
import numpy as np
import os
# import record
import wave

# 生成wavdict，key=wavid，value=wavfile
def gen_wavlist(wavpath):
	wavdict = {}
	labeldict = {}
	for (dirpath, dirnames, filenames) in os.walk(wavpath):
		# print(dirpath,dirnames,filenames)
		for filename in filenames:
			if filename.endswith('.wav'):
				
				filepath = os.sep.join([dirpath, filename])
				# fileid = filename.strip('.wav')
				fileid = filepath
				# print(filepath)

				wavdict[fileid] = filepath
				# label = fileid.split('_')[1]
				label = filepath.split("/")[1]
				# print(label)
			
				labeldict[fileid] = label
	return wavdict, labeldict

# 生成wavdict，key=wavid，value=wavfile
def gen_wavlist2(wavpath):
	wavdict = {}
	labeldict = {}
	for (dirpath, dirnames, filenames) in os.walk(wavpath):
		for filename in filenames:
			if filename.endswith('.wav'):
				filepath = os.sep.join([dirpath, filename])
				fileid = filename.strip('.wav')
				# print(fileid)
				wavdict[fileid] = filepath
				label = fileid.split('_')[1]
				labeldict[fileid] = label
	return wavdict, labeldict

# 特征提取，feat = compute_mfcc(wadict[wavid])
def compute_mfcc(file):
	# print(file)
	fs, audio = wavfile.read(file)
	mfcc_feat = mfcc(audio, samplerate=fs,numcep=13, winlen=0.025, winstep=0.01,nfilt=26, nfft=2048, lowfreq=0, highfreq=None, preemph=0.97)
	d_mfcc_feat = delta(mfcc_feat, 1)
	d_mfcc_feat2 = delta(mfcc_feat, 2)
	# mfcc_feat = mfcc_feat + d_mfcc_feat
	# feature_mfcc = np.hstack((mfcc_feat, d_mfcc_feat, d_mfcc_feat2))
	feature_mfcc = np.hstack((mfcc_feat, d_mfcc_feat))
	# print(mfcc_feat)

	# print(mfcc_feat)
	# feature_mfcc = np.hstack((mfcc_feat))
	# print(feature_mfcc)
	return feature_mfcc
	# return mfcc_feat

class Model():
	def __init__(self, CATEGORY=None, n_comp=1, n_mix = 1, cov_type='full', n_iter=100000):
		super(Model, self).__init__()
		print(CATEGORY)
		self.CATEGORY = CATEGORY
		self.category = len(CATEGORY)
		self.n_comp = n_comp
		self.n_mix = n_mix
		self.cov_type = cov_type
		self.n_iter = n_iter
		# 关键步骤，初始化models，返回特定参数的模型的列表
		self.models = []
		for k in range(self.category):
			# model = hmm.GMMHMM(n_components=self.n_comp, n_mix = self.n_mix, 
			# 					covariance_type=self.cov_type)
			model = hmm.GMMHMM(n_components=self.n_comp, n_mix = self.n_mix,covariance_type=self.cov_type )
			self.models.append(model)

	# 模型训练
	def train(self, wavdict=None, labeldict=None):
		# print("self.CATEGORY",self.CATEGORY)
		# print("labeldict",labeldict)
		# print("wavdict",wavdict)
		for k in range(10):
			subdata = []
			model = self.models[k]
			for x in wavdict:
				if labeldict[x] == self.CATEGORY[k]:
					print("k=",k,wavdict[x])

					mfcc_feat = compute_mfcc(wavdict[x])
					# print(mfcc_feat)
					# print(mfcc_feat)
					model.fit(mfcc_feat)

	# 使用特定的测试集合进行测试
	def test(self, filepath):
		result = []
		for k in range(self.category):
			subre = []
			label = []
			model = self.models[k]
			mfcc_feat = compute_mfcc(filepath)
			# 生成每个数据在当前模型下的得分情况
			re = model.score(mfcc_feat)
			subre.append(re)
			result.append(subre)
		# 选取得分最高的种类
		result = np.vstack(result).argmax(axis=0)
		# 返回种类的类别标签
		result = [self.CATEGORY[label] for label in result]
		# print('识别得到结果：\n',result)
		return result


	def save(self, path="models.pkl"):
		# 利用external joblib保存生成的hmm模型
		joblib.dump(self.models, path)


	def load(self, path="models.pkl"):
		# 导入hmm模型
		self.models = joblib.load(path)


def gen_dataset(wavdict, labeldict):
	nums = len(labeldict)
	shuf_arr = np.arange(nums)
	# print(shuf_arr)
	np.random.shuffle(shuf_arr)
	# print(shuf_arr)
	labelarr = []
	for l in labeldict:
		labelarr.append(l)
	wavdict_test = {}
	labeldict_test = {}
	wavdict_train = {}
	labeldict_train = {}	
	for i in range(int(nums * 0.05)):
		wavdict_test[labelarr[shuf_arr[i]]] = wavdict[labelarr[shuf_arr[i]]]
		labeldict_test[labelarr[shuf_arr[i]]] = labeldict[labelarr[shuf_arr[i]]]
	for k in labeldict:
		if k not in labeldict_test:
			wavdict_train[k] = wavdict[k]
			labeldict_train[k] = labeldict[k]
		
	return wavdict_train, labeldict_train, wavdict_test, labeldict_test




# 准备训练所需数据


if __name__=='__main__':
	CATEGORY = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
	wavdict, labeldict = gen_wavlist("data_zh")
	# wavdict, labeldict = gen_wavlist2('training_data')
	# print("wavdict",wavdict)
	# print("---------------------------")
	# print("labeldict",labeldict)
	wavdict_train, labeldict_train, wavdict_test, labeldict_test = gen_dataset(wavdict, labeldict)
	print("wavdict_train",labeldict_train,len(wavdict_train))
	print("wavdict_test",labeldict_test,len(wavdict_test))

	# wavdict_test = wavdict_train
	# labeldict_test = labeldict_train
    # wavdict, labeldict = gen_wavlist2('data_zh_2')
    # testdict, testlabel = gen_wavlist('test_data')
    # 进行训练
	models = Model(CATEGORY=CATEGORY)
	print("start trainging....")

	models.train(wavdict=wavdict, labeldict=labeldict)
	print("finish trainging....")
	models.save()
	models.load()
	print('test begin!')

	TP = 0
	FP = 0
	for k in wavdict_test:
		wav_path = wavdict_test[k]
		res = models.test(wav_path)[0]
		print(wavdict_test[k],res,labeldict_test[k])
		if res == labeldict_test[k]:
			TP += 1		
		else:
			FP += 1
	print(TP,FP)
	print("acc:",TP/(TP+FP))


	
    # while True:
    #     print('序号1：打开电灯， 序号2：关闭电灯， 序号3：空调开启， 序号4：空调关闭， 序号5：升高温度')
    #     print('序号6：降低温度， 序号7：播放音乐， 序号8：停止播放， 序号9：提升音量， 序号10：降低音量')
    #     a = input('输入1进行语音检测，输入0停止测试,请输入您的选择：')
    #     if a == '1':
    #         r = record.recoder()
    #         print('开始')
    #         r.recoder()
    #         r.savewav("test.wav")
    #         models.test('test.wav')
    #     else:
    #         exit(0)