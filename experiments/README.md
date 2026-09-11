# experiments

这里存放可重复运行的实验入口和参数配置。

推荐原则：

- 一个实验只回答一个问题；
- 所有实验保存统一日志字段；
- 不把控制器公式复制到实验脚本里；
- Task11 起支持批量参数扫描。

建议统一记录：

```text
time
q / qdot
tau_cmd
TCP pose / velocity
TCP wrench
object pose / velocity
reference
controller gains
simulator settings
```

Task13 的 MuJoCo / Isaac 对照必须使用同一套评价指标。
