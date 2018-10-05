


import sys
sys.path.append('..')

import config
from tools import GpuData, RequestData


def check_request(uid, *args):
	requests = RequestData(config.REQUEST_DATA) 
	live = requests.slice({
			'finish' : False,
			'uid':uid})
	return str(live)





