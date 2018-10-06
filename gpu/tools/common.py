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
		#try:
		#	return self.args[cmd](*args, **kwargs)  
		#except:
		#	return 'exception'
		return self.args[cmd](*args, **kwargs) 

	def __getitem__(self, cmd):
		if cmd in self.args.keys():
			return self.args[cmd]
		else:
			raise ValueError('command %s not register.' % cmd)