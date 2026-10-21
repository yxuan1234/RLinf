# Reference OPD through Teacher routing

Reference OPD uses the frozen Base SFT as the routed Teacher for retained
suites. Each sample has exactly one Teacher target and one OPD loss:

```text
L = mean_masked(0.5 * (mu_student - mu_routed_teacher)^2)
```

Teacher and Student predict the ODE step mean at the same stored Student
observation, latent state and flow time. Teacher targets are detached and
cached before Student updates. The loss averages over valid action elements
and selected flow steps. It is an ODE mean-matching surrogate, not an exact
KL between deterministic transitions.

## Retained configurations

- `opd/libero/spatial_opd_objkl`:
  Spatial routes to the S/O/G PPO Teacher; Object routes to frozen Base SFT.
  The historical `objkl` filename refers to reference OPD on Object.
- `opd/libero/libero_sog_plus10_opd_ppo120_teacher_vlm_save50_x5lr`:
  the main consolidation run, with S/O/G routed to their shared PPO Teacher
  and LIBERO-10 routed to its PPO Teacher.

There is no additional Base SFT anchor loss on top of the routed OPD loss.
The reference is simply a frozen Teacher selected for the retained suite.
All recipes collect online Student rollouts.

## 中文说明

Reference OPD 与普通 OPD 使用同一种 loss。区别只在 Teacher 路由：需要保持的
suite 用冻结 Base SFT 当 Teacher，需要学习的 suite 用对应 PPO Teacher。
每条样本只计算一个 OPD loss，不额外叠加第二个 Base SFT loss。

## CPU regression checks

```bash
python -m pytest tests/unit_tests/test_opd_reference_routing.py \
  tests/unit_tests/test_opd_multistep_prefix_cache.py -q
```
