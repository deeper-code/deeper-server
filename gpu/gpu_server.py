import sys
import os
#import argparse
from tools import remote


class Action(object):
	'''Class for register action to special argument.'''
	def __init__(self, name='None'):
		self.name = name
		self.args = {}

	def register(cmd, action):
		if self.args.get(cmd, 'None') != 'None':
			raise ValueError('Command %s is already exists.' % cmd)
		if not callable(action):
			raise ValueError('Action must be callable.')

		self.args[cmd] = action

	def call(cmd, *args, **kwargs):
		return self.args[cmd](*args, **kwargs)




def regest_actions():
	actions = Action(name='cbib')
	#actions.register('check', )
	return actions

def main():
	
	pipe = remote.PipeRemote(itype='server')

	print(pipe.accept())
	#print(args)
	pipe.send(b'i\'m server')
	pipe.clear()

if __name__ == '__main__':
	main()
