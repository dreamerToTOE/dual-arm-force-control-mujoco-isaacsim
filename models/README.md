# models

这里存放本项目明确纳入版本控制的机器人/场景模型说明与轻量配置。

第一阶段优先复用 MuJoCo Menagerie 的 Franka FR3 模型；不要把完整第三方仓库或大型二进制资源直接复制进来。

正式引入第三方模型时需要记录：

- 来源仓库与 commit/tag；
- License；
- 本项目做了哪些修改；
- FR3 joint order、base frame、TCP frame；
- 与 Isaac Sim 中 FR3 模型的差异。

大型外部资源放在本地 `third_party/`，默认由 `.gitignore` 排除。
