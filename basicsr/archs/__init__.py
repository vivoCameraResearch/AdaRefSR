import importlib
from copy import deepcopy
from os import path as osp

from basicsr.utils import get_root_logger, scandir
from basicsr.utils.registry import ARCH_REGISTRY

# import sys
# import os
# import os.path as osp

# # 将项目的根目录加入到 Python 的搜索路径
# sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../..")
# # # 将 `basicsr` 所在目录加入 sys.path

__all__ = ['build_network']

# import sys
# import os

# # 获取当前文件的绝对路径
# current_dir = os.path.dirname(os.path.abspath(__file__))

# # 假设 `basicsr` 位于当前文件的上两级目录中
# project_root = os.path.abspath(os.path.join(current_dir, "../../"))
# if project_root not in sys.path:
#     sys.path.append(project_root)
    
    
# basicsr_root = "工号/code/taoge/S3Diff-main"
# if basicsr_root not in sys.path:
#     sys.path.append(basicsr_root)

# print("Updated sys.path:", sys.path)


# automatically scan and import arch modules for registry
# scan all the files under the 'archs' folder and collect files ending with '_arch.py'
arch_folder = osp.dirname(osp.abspath(__file__))
arch_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(arch_folder) if v.endswith('_arch.py')]
# import all the arch modules
_arch_modules = [importlib.import_module(f'basicsr.archs.{file_name}') for file_name in arch_filenames]


def build_network(opt):
    opt = deepcopy(opt)
    network_type = opt.pop('type')
    net = ARCH_REGISTRY.get(network_type)(**opt)
    logger = get_root_logger()
    logger.info(f'Network [{net.__class__.__name__}] is created.')
    return net
