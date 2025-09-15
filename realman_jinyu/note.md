##

<body name="robot_arm_right" pos="0.6 0 0">
      <inertial pos="-0.000433277 -3.54664e-05 0.0599428" quat="0.997516 0.00259716 0.0221813 0.0668003" mass="0.841071" diaginertia="0.00172808 0.00170955 0.000902378"/>
      <geom type="mesh" rgba="1 1 1 1" mesh="base_link"/>
      <body name="Link1_right" pos="0 0 0.2405">
        <inertial pos="1.22263e-08 0.021108 -0.0251854" quat="0.988851 -0.148906 -2.80074e-05 -0.000243475" mass="0.593563" diaginertia="0.00126614 0.00124677 0.000496264"/>
        <joint name="joint1_right" pos="0 0 0" axis="0 0 1" range="-3.1 3.1" actuatorfrcrange="-60 60"/>
        <geom type="mesh" rgba="1 1 1 1" mesh="link1"/>

inertial 是这个body的物理特性，密度呀，质心位置呀
joint  是指物体可不可以移动，相对于父body type=free的时候指这个物体可以随便移动，六个自由度都可以移动
 <joint name="joint1_right" pos="0 0 0" axis="0 0 1" range="-3.1 3.1" actuatorfrcrange="-60 60"/>

 没写 type 时，默认就是 hinge（转动关节） hinge = 1 DoF 旋转关节：绕 axis="0 0 1" 指定的轴转动；pos 是关节锚点（在该 body 的局部坐标系里）。
 pos="0 0 0"：关节锚点在“该关节所在的 body 的局部坐标系”里的坐标，axis="0 0 1"：关节轴（hinge）或滑动方向（slide）在“该 body 的局部坐标系”里
这个joint只有单自由度，只能绕着父body的z轴转动

geom   是指这个物体是什么物体，这个物体长什么样


<body name="table" pos="0 0 0">   <!-- 高度 0，可改 0.7 等 -->
    <inertial pos="0 0 0" mass="1000" diaginertia="1 1 1"/>
    <geom type="box" size="0.8 0.6 0.02" material="table_mat" pos="0 0 -0.02" condim="3" friction="1 0.005 0.0001" solimp="0.98 0.98 0.001" solref="0.002 2000"  margin="0" />
</body>

小提示：你之前的 adverse 就是 ellipsoid。如果觉得接触太“滑”，可以适当增大 friction 的第一项（滑动摩擦），比如 friction="1.2 0.01 0.001"；如果接触过硬引起抖动，把 solref 的第一项（时间常数）稍微变大些，比如 solref="0.02 1"。
friction="0.5"

geom 的摩擦其实是 3 个数：[滑动, 扭转, 滚动]。

只写一个数时，只设置滑动摩擦=0.5；其余两项用默认值。

想显式全写：例如 friction="1.0 0.005 0.0001"（常见：滑动大、扭转/滚动很小）。

solref="0.01 1"

这是接触/约束的软约束参数：[timeconst, dampratio]。

timeconst（时间常数，秒）：越小越“硬”（更快把穿透/误差纠回）；

dampratio（阻尼比）：1≈临界阻尼（基本不反弹），<1 会有弹跳，>1 更粘更不弹。

所以 0.01 1 = 较硬、无弹跳 的接触。再小可能不稳（抖动/发散）。

补充：接触通常还会配合 solimp="..."（形状/非线性映射），不写就用默认值；大多数场景直接调 solref 就够用了。

geom的type：
<!-- 盒子：实际 0.04 x 0.02 x 0.02 m -->
<geom type="box" size="0.02 0.01 0.01" rgba="1 0 0 1"
      friction="1.0 0.005 0.0001" solref="0.01 1"/>

<!-- 椭球：X/Y/Z 半径分别 1cm、1.5cm、0.5cm -->
<geom type="ellipsoid" size="0.01 0.015 0.005" rgba="0 0 1 1"
      friction="1.0 0.005 0.0001" solref="0.01 1"/>


geom 的 type（几何体/碰撞形状）

可选：plane, hfield, sphere, capsule, ellipsoid, cylinder, box, mesh，以及 sdf（需要插件）。
尺寸参数因类型不同而异，例如：box 的 size="sx sy sz" 是半尺寸；sphere 只要半径；capsule/cylinder 是半径和半高；ellipsoid 是三轴半径；mesh/hfield 忽略 size 用资产自身尺寸。
mujoco.readthedocs.io

补充：site 的几何类型是子集，只支持 sphere/capsule/ellipsoid/cylinder/box。
mujoco.readthedocs.io


joint 的 type（关节/自由度）

可选只有 4 种：free, ball, slide, hinge；不写时默认就是 hinge。

free：6 自由度（3移+3转），只能放在世界子 body；不能限位。

ball：3 转不移（单位四元数表示）。

slide：沿给定轴平移 1 自由度。

hinge：绕给定轴转动 1 自由度。
mujoco.readthedocs.io
+1

如果你说下要建的物体/约束，我可以直接给出对应 geom 与 joint 的最小可用片段（含推荐 size/solref/solimp/friction）。



<joint name="adverse_joint" type="free" damping="1e8"/>