# HMM_Digital_Voice_Recognition
基于HMM与MFCC特征进行数字0-9的语音识别，HMM，隐马尔可夫，GMMHMM，MFCC，语音识别，sklearn，Digital Voice Recognition。

## Preinstallation
```
 conda create -n HMM python=3.6 numpy pyaudio scipy hmmlearn scipy #也可以使用pip
 conda activate HMM
 pip install -r requirements.txt
```

数据链接: https://pan.baidu.com/s/124TiAs8m7Ioa2_3dUrxGSg 提取码: xsfe

以下命令假设下载数据至/tmp/dataset.zip


## Launch the script
```
  git clone https://github.com/zhengyima/HMM_Digital_Voice_Recognition/ HMM_DVR
  cd HMM_DVR
  unzip /tmp/dataset.zip -d ./  # dataset.zip是从百度网盘下载的数据
  python hmm_gmm.py
  
```

## Links

[GMM实现](https://github.com/zhengyima/GMM_Digital_Voice_Recognition)

[DTW实现](https://github.com/zhengyima/DTW_Digital_Voice_Recognition)
