
import pandas as pd
import os
import sys
sys.path.append('..')

from gpu import config

class DataBase(object):
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




class GpuData(DataBase):
	def __init__(self, file, reset=False):
		super(GpuData, self).__init__(file, reset)

		self.name = 'gpu database'
		if not os.exists(file) and reset == False:
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
			self.df =  pandas.read_csv(file)
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


class UserData(DataBase):
	def __init__(self, file, reset=False):
		self.name = 'user information database'
		self.file = file
		if not os.exists(file) and reset == False:
			raise ValueError('Database file %s not exists.' % file)


		self.columns = ['uid', 'gid', 'uuid', 'name']


		if not reset:
			self.df =  pandas.read_csv(file)
		else:
			self.reset()

	def reset(self):
		if os.path.exists(self.file):
			os.remove(self.file)
		self.df = pd.DataFrame(columns=self.columns)
		# get all user information.
		


		self.local()









