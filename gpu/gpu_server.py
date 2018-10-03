import sys
import os
#import argparse
from tools import remote
from actions import check
from tools import GpuData, RequestData
import config


class Action(object):
	'''Class for register action to special argument.'''
	def __init__(self, name='None'):
		self.name = name
		self.args = {}

	def register(self, cmd, action):
		if self.args.get(cmd, 'None') != 'None':
			raise ValueError('Command %s is already exists.' % cmd)
		if not callable(action):
			raise ValueError('Action must be callable.')

		self.args[cmd] = action

	def call(self, cmd, *args, **kwargs):
		return self.args[cmd](*args, **kwargs)




def regest_actions():
	actions = Action(name='cbib')
	actions.register('check',  check.check_request)
	return actions

def main():
	
	pipe = remote.PipeRemote(itype='server')
	actions = regest_actions()

	requests = RequestData(config.REQUEST_DATA, reset=True) 



	cmd = pipe.accept()
	print(cmd)
	if cmd == 'check':
		print('True')
		pipe.send(actions.call('check', uid=1001))
	#print(args)
	#pipe.send(b'i\'m server')
	#pipe.clear()

if __name__ == '__main__':
	main()




