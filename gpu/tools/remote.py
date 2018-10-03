
import sys
sys.path.append('..')

from gpu import config
import os


class Remote(object):
	'''Python Process call.'''
	def __init__(self, itype='server'):
		self.name = 'basic'
		if itype not in ['server', 'client']:
			raise ValueError('itype must be in [server | client]')
		self.itype = itype

	def prepare(self):
		raise NotImplementedError('subclass is allowed to instant.')

	def send(self, msg):
		raise NotImplementedError('subclass is allowed to instant.')

	def accept(self):
		raise NotImplementedError('subclass is allowed to instant.')

	def clear(self):
		raise NotImplementedError('subclass is allowed to instant.')



""" Communication Inter Process with pipe"""
class PipeRemote(Remote):
	def __init__(self, itype):
		self.name = 'pipe'
		self.itype = itype
		self.prepare()


	def prepare(self):
		if self.itype == 'server':
			if os.path.exists(config.FIFO_IN):
				os.remove(config.FIFO_IN)
			if os.path.exists(config.FIFO_OUT):
				os.remove(config.FIFO_OUT)

			os.mkfifo(config.FIFO_IN,  0o644)
			os.mkfifo(config.FIFO_OUT, 0o644)

			self.rf = os.open(config.FIFO_IN, os.O_RDONLY)
			self.wf = os.open(config.FIFO_OUT, os.O_SYNC | os.O_CREAT | os.O_RDWR)

		else:
			if not os.path.exists(config.FIFO_IN) or \
			   not os.path.exists(config.FIFO_OUT):
			   raise IOError('FIFO file not exists. You may run sever frist.')

			self.rf = os.open(config.FIFO_OUT, os.O_SYNC | os.O_RDWR)
			self.wf = os.open(config.FIFO_IN, os.O_SYNC | os.O_CREAT | os.O_RDWR)

	def send(self, msg):
		os.write(self.wf, msg)


	def accept(self):
		return os.read(self.rf, 2048)

	def clear(self):
		os.close(self.rf)
		os.close(self.wf)
		if os.path.exists(config.FIFO_IN):
			os.remove(config.FIFO_IN)
		if os.path.exists(config.FIFO_OUT):
			os.remove(config.FIFO_OUT)












