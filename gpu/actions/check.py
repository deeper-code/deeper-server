


import sys
sys.path.append('..')

import config
from tools import GpuData, RequestData
from tools import remote


def server_check_request(uid, *args):
	requests = RequestData(config.REQUEST_DATA) 
	live = requests.slice({
			'finish' : False,
			'uid':uid})
	return str(live)



def client_check_request(uid, *args):
	return '%d|check|none' % uid








