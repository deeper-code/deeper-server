

import sys
sys.path.append('..')

import config
from tools import GpuData, RequestData, GPUs



def get_lock(uid):
	''' Can this user(uid)  request gpu? 
	'''
	requests = RequestData(config.REQUEST_DATA) 
	gpus     = GPUs()


	live = requests.slice({
				'finish' : False,
				'uid':uid})

	# @codewang 2018.10.4 V0.0.1 : Those cases cannot request GPUs.
	# requests.columns = ['rid','uid', 'uuid', 'name', 'start', 'end', 'gpu_list', 'group_id', 'finish']
	# gpus.columns = ['nr', 'status', 'onwer', 'start', 'end', 'why']
	# 
	# 1. Already request a low of gpus that not in using.
	# 2. It's time to push back his(her) GPUs, but he(she) didn't.
	# 3. In blacklist 

	for l in live.index:
		gpu_list = live['gpu_list'].loc[l].split()
		for nr in gpu_list:
			total, used, free, processes = gpus[nr].data()
			if len(processes) == 0:
				# No process
				return False, 'The GPU-%d you requested is not used, so you can not continue to request another gpus.' % nr
	return True,''

			

def gpu_get(uid, gpu_list):
	# check usre infomation.
	requests = RequestData(config.REQUEST_DATA) 
	gpudata  = GpuData(config.GPU_DATA)

	live = requests.slice({
			'finish' : False,
			'uid':uid})

	lock, ret = get_lock(uid)
	if not lock:
		return ret

	# check index
	_list = []
	for _inx in gpu_list:
		inx = int(_inx)
		if  0 <= inx < config.NR_GPU:
			_list.append(inx)
		else:
			return 'error, %d is not allowed' % inx

	# check gpu status
	for inx in _list:
		if gpudata.df['status'].iloc[inx] != 'free':
			return 'GPU-%d you requested is not free. using gpu -l to have a checking.' % inx









