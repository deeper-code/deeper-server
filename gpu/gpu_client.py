
import sys
import os
import argparse
from tools import remote, Action
from actions import check, request
from subprocess import getstatusoutput


def parse_parames():
	parse = argparse.ArgumentParser()
	request_group = parse.add_mutually_exclusive_group()
	request_group.add_argument('-l', '--list',   action='store_true', help='查询GPU状态', default=False)
	request_group.add_argument('-c', '--check',   action='store_true', help='查询已申请GPU资源', default=False)
	request_group.add_argument('-v', '--version', action='store_true', help='查询当前版本号', default=False)
	request_group.add_argument('-g', '--get', type=int, help='申请指定编号GPU,例如: gpu -g 1 5 6', 
											  nargs='*', choices=range(10), default=False)
	request_group.add_argument('-r', '--request', type=int, help='申请指定数量GPU(系统分配)， 例如: gpu -r 3', 
											  nargs=1, choices=range(1,5), default=False)
	request_group.add_argument('-p', '--push', type=int, help='释放指定编号GPU, 例如: gpu -p 1 5 6', 
											  nargs='*', choices=range(10), default=False)
	request_group.add_argument('-d', '--delete', type=int, help='释放指定组编号GPU', nargs=1, default=False)
	request_group.add_argument('-a', '--ask', action='store_true', help='申请使用公共账号', default=False)
	request_group.add_argument('-b', '--back', action='store_true', help='归还公共账号', default=False)

	return parse.parse_args()


# method
# def method(uid(int), args(string))
# return a string with format ``uid|command|arg1|arg2|...`` for send to server 
# atleast one arg needed.
def regest_actions():
	actions = Action(name='cbib')
	actions.register('check', check.client_check_request)
	actions.register('get', request.client_gpu_get)
	return actions



def main():
	# parsing arguments from console.
	args = vars(parse_parames())

	# to create remote-handel <using pipe> 
	pipe = remote.PipeRemote(itype='client')

	# register actions 
	actions = regest_actions()

	status, uid = getstatusoutput('id -u')
	uid = int(uid) 

	for cmd, arg_list in args.items():
		if arg_list:
			msg = actions.call(cmd, uid, arg_list)
			print('send: ', msg)
			pipe.send(msg)
			replay = pipe.accept()
			print(replay)

	pipe.clear()
	exit()
	
if __name__ == '__main__':
	main()









