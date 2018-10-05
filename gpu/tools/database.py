
import pandas as pd
import os
import sys
sys.path.append('..')

from subprocess import getstatusoutput

import pandas as pd
from gpu import config

class DataBase(object):
	""" An abstract class for datasets.
	"""

	def __init__(self, file, reset=False):
		self.name = 'None'
		self.file = file
		self.columns = []

	def set(nr, key, value, save=True):
		self.df[key].iloc[nr] = value
		if save:
			self.local()

	def local(self):
		self.df.to_csv(self.file, index=None, header=True)

    
	def __getitem__(self, index):
		return self.df.iloc[index].values


	def __len__(self):
		return self.df.shape[0]


	def __new__(cls, *args, **kwargs):
		"""singleton mode
		"""
		if not hasattr(cls, '_instance'):
			cls._instance = super().__new__(cls)
		return cls._instance


class _GpuCore(object):
	""" GPU information for each GPU.
	include gpu memary in using/free/total
	and each process on this gpu
	"""
	def __init__(self, nr=-1):
		self.nr = nr
		# procs --> precesses
		# for each processing:
		# pid  : this processing pid
		# name : command line
		# gpu_mem :
		self.procs = []
		self.total = 0 
		self.used  = 0
		self.free  = 0


	def data(self):
		total = int(self.total.split()[0].strip())
		used  = int(self.used.split()[0].strip())
		free  = int(self.free.split()[0].strip())

		processes = [[x['pid'], x['name'], 
					  int(x['gpu_mem'].split()[0].strip())  ]  for x in self.procs]

		return total, used, free, processes

	def updata(self):
		''' console command : sudo nvidia-smi -q -i 5 -d PIDS,MEMORY
			output:
			GPU 00000000:0B:00.0
			    FB Memory Usage
			        Total                       : 11172 MiB
			        Used                        : 10795 MiB
			        Free                        : 377 MiB
			    Processes
			        Process ID                  : 44070
			            Type                    : C
			            Name                    : /home/ljg/anaconda3/envs/tensorflow/bin/python
			            Used GPU Memory         : 10785 MiB

		'''
		stat, output = getstatusoutput('nvidia-smi -q -i %d -d PIDS,MEMORY' % self.nr)
		if stat == 0:
			output = [x.strip() for x in  output.strip().split('/n')]
			mem_inx  = output.index('FB Memory Usage')
			# memeary
			self.total = output[mem_inx+1].split(':')[-1].strip()  # 11172 MiB
			self.used  = output[mem_inx+2].split(':')[-1].strip()
			self.free  = output[mem_inx+3].split(':')[-1].strip()

			# processes
			try:
				proc_inx = output.index('Processes')
				for inx in range(proc_inx+1, len(output), 4):
					# find a process
					if output[inx].split(':')[0].strip() == 'Process ID':
						P = {}
						P['pid']     = int(output[inx].split(':')[1].strip())
						P['name']    = output[inx+2].split(':')[1].strip()
						P['gpu_mem'] = output[inx+3].split(':')[1].strip()
						self.procs.append(P)
			except:
				# No processes
				pass

			# updata GpuData
			pass



class GPUs(Database):
	""" runtime GPU status.
	"""
	def __init__(self):
		super(GPUs, self).__init__(None, reset=False)
		
		self._gpus = []
		for i in range(config.NR_GPU):
			gpu = _GpuCore(nr=i)
			gpu.update()
			self._gpus.append(gpu)

	def __getitem__(self, inx):
		return self._gpus[inx]

	def updata(self):
		for i in range(config.NR_GPU):
			self._gpus[i].update()



class GpuData(DataBase):
	def __init__(self, file, reset=False):
		super(GpuData, self).__init__(file, reset)

		self.name = 'gpu database'
		if not os.path.exists(file) and reset == False:
			raise ValueError('Database file %s not exists.' % file)

		self.file = file
		# GPU status
		#  nr : Nomber of gpu range from 0 to 9
		#  status : free/requested/using/release/other
		#  onwer  : who is using this gpu.
		#  start. : time of requesting this gpu
		#  end.   : time of auto free this gpu
		#  why.   : why auto free this gpu, [requested, released]
		# 
		self.columns = ['nr', 'status', 'onwer', 'start', 'end', 'why']

		if not reset:
			self.df =  pd.read_csv(file)
		else:
			self.reset()

	def reset(self):
		""" Reset gpu database """
		if os.path.exists(self.file):
			os.remove(self.file)
		self.df = pd.DataFrame(columns=self.columns)
		self.df['nr']     = range(config.NR_GPU)
		self.df['status'] = 'free'
		self.df['onwer']  = 'root'
		self.df['start']  = None 
		self.df['end']    = None
		self.df['why']    = None

		self.local()


class RequestData(DataBase):
	def __init__(self, file, reset=False):

		super(RequestData, self).__init__(file, reset)
		self.name = 'user information database'
		self.file = file
		if not os.path.exists(file) and reset == False:
			raise ValueError('Database file %s not exists.' % file)


		# rid : request id
		# uid  user id
		# uuid user name simply
		# name user name
		# start : when this user request gpu
		# using : when this user start to using gpu
		# release  : when the processes are  stoped
		# end : when this user push back those gpus.
		# group_id : id of request.
		# finish : pass
		self.columns = ['rid','uid', 'uuid', 'name', 'start', 'end', 'gpu_list', 'group_id', 'finish']
		if not reset:
			self.df =  pd.read_csv(file)
		else:
			self.reset()


	def slice(self, limits):
		_lmt = True
		for k,v in limits.items():
			_lmt = _lmt &  (self.df[k] == v)
		return self.df[_lmt]

	def reset(self):
		if os.path.exists(self.file):
			os.remove(self.file)
		self.df = pd.DataFrame(columns=self.columns)
		pass
		self.local()


















