

from .common import Action
from .database import GpuData, RequestData, GPUs
from .remote import PipeRemote


# GpuData : GPU status (pandas.DataFrame)      
#  
#  nr : Nomber of gpu range from 0 to 9
#  status : free/requested/using/release/other
#  onwer  : who is using this gpu.
#  start. : time of requesting this gpu
#  end.   : time of auto free this gpu
#  why.   : why auto free this gpu, [requested, released]
# 


# RequestData : Requests record. (pandas.DataFrame)
#
#  rid : request id
#  uid  user id
#  uuid user name simply
#  name user name
#  start : when this user request gpu
#  using : when this user start to using gpu
#  release  : when the processes are  stoped
#  end : when this user push back those gpus.
#  gpu_list
#  group_id : id of request.
#  finish : pass


# GPUs : vector of _GpuCore (runtime GPUs status)
#    _GpuCore : status of one GPU.
#         total : total FB-memary 
#         used  : pass
#         free  : pass
#         procs : list of process-status-dictionary
#              
#   process-status-dictionary:
#         pid     : pass
#         name    : pass
#         gpu_mem : pass
#
# 
#   all of those information will refresh when calling GPUs.updata()  

