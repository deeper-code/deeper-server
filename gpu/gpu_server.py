import sys
import os
#import argparse
from tools import remote
from actions import check
from tools import GpuData, RequestData
import config



# method :
# 客户端发来 uid|cmd|args
# 所以每个功能函数需要接受至少参数
# function(uid, arg1, ...)
# 或者
# function(uid, *args)
# 
# 返回值必须是字符串

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
		try:
			return self.args[cmd](*args, **kwargs)
		except:
			return 'exception'


	def __getitem__(self, cmd):
		if cmd in self.args.keys():
			return self.args[cmd]
		else:
			raise ValueError('command %s not register.' % cmd)


def regest_actions():
	actions = Action(name='cbib')
	actions.register('check',  check.check_request)
	return actions


def main():
	# to created remote-handel <using pipe>
	pipe = remote.PipeRemote(itype='server')

	# regeste all actions
	actions = regest_actions()

	# accept commands from client.
	while True:
		buf = pipe.accept()
		# parse command <format : uid|command|args>
		uid, cmd, args = buf.split('|')

		# call 
		replay = actions.call(cmd, uid, *args)
		# send replay
		pipe.send(replay)

if __name__ == '__main__':
	main()




