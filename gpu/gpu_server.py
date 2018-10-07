import sys
import os
#import argparse
from tools import remote, Action
from actions import check, request
from tools import GpuData, RequestData
import config

import argparse

# method :
# 1. def method(uid(int), *args)  
# 2. return a string for replay to client
def regest_actions():
	actions = Action(name='cbib')
	actions.register('check',  check.server_check_request)
	actions.register('get', request.server_gpu_get)
	return actions


def reset():
	requests = RequestData(config.REQUEST_DATA, reset=True)
	gpudata  = GpuData(config.GPU_DATA, reset=True)


def main():
	
	parse = argparse.ArgumentParser()
	parse.add_argument('--reset',   action='store_true', help='重置所有数据库', default=False)
	args = parse.parse_args()

	if args.reset == True:
		reset()

	# to created remote-handel <using pipe>
	pipe = remote.PipeRemote(itype='server')

	# regeste all actions
	actions = regest_actions()

	# accept commands from client.
	while True:
		buf = pipe.accept()  # "1001|check"
		# parse command <format : uid|command|args>
		try:
			print('recved : ', buf)
			uid, cmd, *args = buf.split('|')
			# call 
			replay = actions.call(cmd, int(uid), args)
			
			# send replay
			pipe.send(replay)
			
			# Log
			pass
		except Exception as e:
			print('error, recved: ', buf)
			print(e)
			pass
	pipe.clear()

if __name__ == '__main__':
	main()




