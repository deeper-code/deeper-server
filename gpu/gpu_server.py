import sys
import os
#import argparse
from tools import remote, Action
from actions import check, request
from tools import GpuData, RequestData
import config



# method :
# 1. def method(uid(int), *args)  
# 2. return a string for replay to client
def regest_actions():
	actions = Action(name='cbib')
	actions.register('check',  check.server_check_request)
	actions.register('get', request.server_gpu_get)
	return actions


def main():
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
			replay = actions.call(cmd, uid, args)
			
			# send replay
			pipe.send(replay)
			
			# Log
			pass
		except:
			#print('error, recved: ', buf)
			pass
	pipe.clear()

if __name__ == '__main__':
	main()




