# 注册一定要放在包的 __init__.py 里，这样 import mujoco_learning 就会执行
from gymnasium.envs.registration import register

print("[realman_jinyu] __init__ executed from:", __file__)  # 诊断用

# 有命名空间版本（你要用 `mujoco_learning/xxx` 这种 id 时，必须先声明命名空间）


register(
    id="realman_jinyu/realman-aloha-v1",
    entry_point="realman_jinyu.env.sim_env:RealmanEnv",  
    nondeterministic=True,
)
register(
    id="realman_jinyu/hook-package-v1",
    entry_point="realman_jinyu.env.task_hang_package_env:HookPackageEnv",  
    nondeterministic=True,
)
register(
    id="realman_jinyu/put-cube-v1",
    entry_point="realman_jinyu.env.task_put_cube_env:TubePutEnv",  
    nondeterministic=True,
)