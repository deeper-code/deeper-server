
import sys
import os
import argparse
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



def parse_parames():
	parse = argparse.ArgumentParser()
	request_group = parse.add_mutually_exclusive_group()
	parse.add_argument('-l', '--list',   action='store_true', help='查询GPU状态')
	parse.add_argument('-c', '--check',   action='store_true', help='查询已申请GPU资源')
	parse.add_argument('-v', '--version', action='store_true', help='查询当前版本号')
	request_group.add_argument('-g', '--get', type=int, help='申请指定编号GPU,例如: gpu -g 1 5 6', 
											  nargs='*', choices=range(10))
	request_group.add_argument('-r', '--request', type=int, help='申请指定数量GPU(系统分配)， 例如: gpu -r 3', 
											  nargs=1, choices=range(1,5))
	request_group.add_argument('-p', '--push', type=int, help='释放指定编号GPU, 例如: gpu -p 1 5 6', 
											  nargs='*', choices=range(10))
	request_group.add_argument('-d', '--delete', type=int, help='释放指定组编号GPU', nargs=1)
	request_group.add_argument('-a', '--ask', action='store_true', help='申请使用公共账号')
	request_group.add_argument('-b', '--back', action='store_true', help='归还公共账号')

	return parse.parse_args()





def regest_actions():
	actions = Action(name='cbib')
	#actions.register('check', )
	actions.register('get', xxx)
	return actions

def main():
	# parsing arguments from console.
	args = parse_parames()

	# to create remote-handel <using pipe> 
	pipe = remote.PipeRemote(itype='client')

	# test to request gpu-1 gpu-2 gpu-8 
	pipe.send("1001|get|1|2|8")
	print(pipe.accept())


	# test check my requestions.
	pipe.send("1001|check|none")
	print(pipe.accept())
	
	# clear 
	pipe.clear()

if __name__ == '__main__':
	main()










