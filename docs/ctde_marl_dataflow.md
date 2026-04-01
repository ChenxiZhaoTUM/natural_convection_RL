# CTDE And MARL Dataflow

本文整理当前仓库中 `marl` 与 `CTDE` 两套实现的数据流和代码模块关系，便于对照理解：

- `zcx_2d_NCCV_2phase_periodic_python/bin/drl/drl_tianshou_training/sac_multi.py`
- `zcx_2d_NCCV_2phase_periodic_python/bin/drl/drl_gym_environments/gym_env_nc/marl/NCPseudoEnv.py`
- `zcx_2d_NCCV_2phase_periodic_python/bin/drl/drl_gym_environments/gym_env_nc/marl/wrapper.py`
- `zcx_2d_NCCV_2phase_periodic_python/bin/drl/drl_gym_environments/gym_env_nc/CTDE/train_masac_CTDE.py`
- `zcx_2d_NCCV_2phase_periodic_python/bin/drl/drl_gym_environments/gym_env_nc/CTDE/NCJointEnv.py`
- `zcx_2d_NCCV_2phase_periodic_python/bin/drl/drl_gym_environments/gym_env_nc/CTDE/masac_CTDE.py`

## MARL 数据流

当前 `marl` 本质上是“共享 actor + 局部 critic”的参数共享多智能体训练。

```text
CFD solver
  -> full flow field / gen_flux / local_flux / gen_ke
  -> wrapper 同步给 n_seg 个 pseudo-env

对每个 segment i:
  local obs o_i
    -> shared actor pi(a_i | o_i)
    -> action a_i

所有 a_i
  -> 合成 joint control
  -> CFD 前进一步

CFD 输出
  -> 每个 pseudo-env 计算自己的 reward r_i
     r_i = f(gen_flux, local_flux[i], gen_ke)

训练时每个 pseudo-env 单独进 buffer:
  (o_i, a_i, r_i, o_i')

critic 学的是:
  Q_i(o_i, a_i)
```

关键点：

- actor 是共享的
- critic 不是 joint critic，而是普通单智能体 critic
- reward 里虽然含有 global 量，但 critic 输入不是 global / joint 信息

## 当前 CTDE 数据流

当前 `CTDE` 是“每个 agent 一个 actor + centralized twin critic”的 joint 训练。

```text
joint env reset
  -> CFD solver
  -> full field
  -> 切成 n_seg 个 local patch
  -> joint obs = [o_1, o_2, ..., o_n]

对每个 agent i:
  actor_i(a_i | o_i)
  -> 得到 joint action a = [a_1, ..., a_n]

joint action
  -> 映射成 segment temperatures
  -> CFD 前进一步

CFD 输出
  -> gen_flux / local_flux[1..n] / gen_ke
  -> 先算每段 shaped reward:
     r_1, r_2, ..., r_n
  -> 再取 mean(r_1..r_n) 作为共享 reward r

joint buffer 存:
  ([o_1..o_n], [a_1..a_n], r, [o_1'..o_n'], done)

centralized twin critic 学的是:
  Q1(o_1..o_n, a_1..a_n)
  Q2(o_1..o_n, a_1..a_n)
```

关键点：

- actor 不共享，当前是 `n_seg` 套独立 actor
- critic 是 centralized twin critic
- replay buffer 存的是 joint transition
- obs 目前保留为每段 local patch
- reward 设计已经对齐 `marl` 的 flux / KE shaping 思路

## 共享 Actor + Centralized Critic 的 CTDE

如果后续改成“共享 actor + centralized critic”，环境侧几乎可以不变，主要改算法文件。

```text
joint env reset
  -> CFD solver
  -> full field
  -> 切成 n_seg 个 local patch
  -> joint obs = [o_1, o_2, ..., o_n]

对每个 agent i:
  shared actor pi(a_i | o_i)
  -> 得到 a_i

joint action a = [a_1, ..., a_n]
  -> 映射成 segment temperatures
  -> CFD 前进一步

CFD 输出
  -> gen_flux / local_flux[1..n] / gen_ke
  -> reward 可以继续用当前 CTDE 的共享 reward
     或保留每段 reward 向量供训练使用

joint buffer 存:
  ([o_1..o_n], [a_1..a_n], r 或 r_vec, [o_1'..o_n'], done)

centralized twin critic:
  Q1(o_1..o_n, a_1..a_n)
  Q2(o_1..o_n, a_1..a_n)

shared actor update:
  shared_actor 作用在所有 o_i
  -> 产生 joint action
  -> centralized critic 给 joint 价值
  -> 梯度回到同一套 actor 参数
```

这个版本兼顾两点：

- 保留 `marl` 的共享策略优势
- 保留 `CTDE` 的 centralized training 优势

## 代码层面的模块图

### 1. 当前 MARL

```text
[sac_multi.py]
  - 创建 SubprocVectorEnv
  - 创建 shared actor
  - 创建单智能体 twin critic
  - 用 Tianshou SACPolicy 训练

        | make_env(...)
        v

[NCPseudoEnv.py]
  - 每个 segment 一个 pseudo-env
  - reset/step 时通过 Wrapper 跟别的 pseudo-env 同步
  - 产出:
    obs_i, reward_i, done_i

        | 调用
        v

[wrapper.py]
  - 收集所有 segment 的 action
  - leader 跑一次 CFD
  - 写 result 文件
  - followers 读取同一份 result

        | 调用
        v

[pybind solver]
  - set_segment_temperatures(...)
  - run_case(...)
  - get_global_heat_flux()
  - get_local_phi_flux(i)
  - get_global_kinetic_energy()
  - get_local_velocity(...)
  - get_local_temperature(...)
```

训练流：

```text
o_i
 -> shared actor
 -> a_i
 -> NCPseudoEnv / Wrapper 合并成 joint action
 -> CFD
 -> result
 -> 每个 pseudo-env 各自算 reward_i
 -> 单智能体 replay / 单智能体 critic 更新
```

### 2. 当前 CTDE

```text
[train_masac_CTDE.py]
  - 创建多个 NCJointEnv
  - 创建 MASACCTDE
  - 创建 JointReplayBuffer
  - 训练循环:
    joint obs -> joint action -> env.step -> joint buffer -> update

        | make_env(...)
        v

[NCJointEnv.py]
  - 一个 CFD group 对应一个 joint env
  - reset():
    baseline warmup -> restart -> joint obs
  - step():
    joint action -> seg_temps -> CFD
    -> joint obs'
    -> shared reward
  - obs 形式:
    [o_1, o_2, ..., o_n]
  - reward 形式:
    mean(shaped_rewards_all_segments)

        | 调用
        v

[pybind solver]
  - 同上
```

算法侧：

```text
[masac_CTDE.py]
  - actor_1 ... actor_n
  - centralized critic Q1 / Q2
  - target critic Q1_t / Q2_t
  - JointReplayBuffer

joint obs [o_1..o_n]
  -> actor_1(o_1), actor_2(o_2), ..., actor_n(o_n)
  -> joint action [a_1..a_n]
  -> env

buffer:
  ([o_1..o_n], [a_1..a_n], r, [o_1'..o_n'], done)

critic:
  Q1 / Q2([o_1..o_n], [a_1..a_n])

actor update:
  所有 actor 一起采样
  -> 拼 joint action
  -> centralized critic 评估
```

### 3. 共享 Actor + Centralized Critic 的 CTDE

环境侧几乎可以不改，还是：

```text
[train_masac_CTDE.py]
    |
    v
[NCJointEnv.py]
    |
    v
[pybind solver]
```

核心变化在算法文件：

```text
[shared_masac_ctde.py 或 masac_CTDE.py]
  - 一个 shared actor pi(a | o)
  - centralized twin critic Q1 / Q2
  - target critics
  - JointReplayBuffer
```

训练流：

```text
joint obs = [o_1, o_2, ..., o_n]

for each i:
  a_i = shared_actor(o_i)

joint action = [a_1, ..., a_n]
  -> NCJointEnv.step(joint action)
  -> reward, next_joint_obs

JointReplayBuffer:
  ([o_1..o_n], [a_1..a_n], r, [o_1'..o_n'], done)

centralized critic:
  Q1(o_1..o_n, a_1..a_n)
  Q2(o_1..o_n, a_1..a_n)

shared actor update:
  shared_actor 作用在所有 o_i
  -> 产生 joint action
  -> centralized critic 给一个 joint 价值
  -> 梯度回到同一套 actor 参数
```

## 三种结构对比

```text
当前 marl:
  shared actor         = 是
  centralized critic   = 否
  joint replay         = 否

当前 CTDE:
  shared actor         = 否
  centralized critic   = 是
  joint replay         = 是

shared-actor CTDE:
  shared actor         = 是
  centralized critic   = 是
  joint replay         = 是
```

## 当前结论

- `marl` 更像“共享 actor 的独立学习”
- 当前 `CTDE` 更像“独立 actor 的集中式训练”
- 如果问题满足“段间近似对称 + 全局耦合强”，那么“共享 actor + centralized critic”的 CTDE 往往是更自然的折中方案
