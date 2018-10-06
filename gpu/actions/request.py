

import sys
sys.path.append('..')

import config
from tools import GpuData, RequestData, GPUs
import time


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

	# 1.  1 3 5   request 
	# 2.  2 4 6   using 
	#       

	for l in live.index:
		gpu_list = live['gpu_list'].loc[l].split()
		for nr in gpu_list:
			total, used, free, processes = gpus[nr].data()
			if len(processes) == 0:
				# No process
				return False, 'The GPU-%d you requested is not used, so you can not continue to request another gpus.' % nr
	return True,''


def request_gpu_list(uid, gpu_list): 
	''' Request GPUs in gpu_list and had already checked.
	Tasks:
		1. take a record to database.
		2. add this user to user group that those gpus in.
		TODO: 3. take a record to log file.
	'''
	requests = RequestData(config.REQUEST_DATA)
	gpudata  = GpuData(config.GPU_DATA)

	# 1. requests
	#uuid, name = get_uuid(uid)
	live = requests.slice({
				'finish' : False,
				'uid':uid})

	requests.df.append({
			'rid'   : requests.df.shape[0] + 1,
			'uid'   : uid,
			'uuid'  : uuid,
			'name'  : name,
			'start'    : time.time(),
			'using'    : None,
			'end'      : None,
			'release'  : None,
			'gpu_list' : ' '.join([str(x) for x in gpu_list]) ,  
			'group_id' : live.shape[0],
			'finish'   : False
		})


	for nr in gpu_list:
		# update database.
		#gpudata.df.iloc[nr] 
		gpudata.df['status'].iloc[nr] = requested
		gpudata.df['onwer'].iloc[nr]  = uid
		gpudata.df['start'].iloc[nr]  = time.time()
		gpudata.df['end'].iloc[nr]    = time.time() + 3600.0
		gpudata.df['why'].iloc[nr]    = 'requested'

		# 
		# 添加此用户到对应的GPU分组中。
	return 'success'



# user : gpu -get 1 5 7
# client :  1001|get|1|5|7
# server :  gpu_get(1001, [1,5,7])

def server_gpu_get(uid, gpu_list):
	# check usre infomation.
	gpudata  = GpuData(config.GPU_DATA)

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

	# request GPUs
	request_gpu_list(uid, _list)



def client_gpu_get(uid, gpu_list):
	''' gpu_list = `[1, 2, 3]` 
	target : uid|command|1|2|3
	'''
	ret = '%d|get|' % uid + '|'.join([str(x) for x in gpu_list])
	return ret


