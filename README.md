本项目**针对泡姆棋的修改**：  
**注意力模块（三消特性）**  
>本项目特色应该是于2507左右尝试抛弃所有卷积层，全部用注意力层代替
>粗略测试大约1/3的参数量可以达到残差块的效果
>260106看到PoPE和mHC的论文，一时兴起直接改了进去，发现效果极佳，于是直接覆盖旧版本，顺便使用Muon优化器，ai帮忙改的真快  
>260108权重确实强了一些但也没啥质变
```
https://arxiv.org/abs/2512.24880v1
https://arxiv.org/abs/2509.10534v2
https://arxiv.org/abs/2502.16982
```

**分阶段搜索（类似于Playout Cap Randomization针对泡姆棋的改进）**  
本项目没有明确的elo曲线，有使用与历史权重和友情项目（Shenyqqq）的权重的对打测试和loss对比以确认技术改进的有效性  
>tips:  
>实际上只需要对比在一定量的棋谱下，什么架构能用最小的参数量达到饱和拟合效果即可  
>在一定量的棋谱下，loss是存在下限的（即棋谱质量的上限）  
>用上述方法对比的架构，在同参数量下，基本都是对打测试更强的架构

同时也主要使用了katago与alphazero等项目的优化技术  
主要参考资料：
```
https://katagotraining.org/
https://arxiv.org/pdf/1902.10565
https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md
```

提供一个exe文件以便于体验：
>CPU的ONNX引擎，存在一些字体不清晰的bug，不明白打包前和打包后为什么有差异，但至少不影响使用，就先不修了
```
和多数玩的熟练的人相近的版本
链接: https://pan.baidu.com/s/18wJM3pfKcVaThANVlOH5IQ?pwd=2333
提取码: 2333
同参数量同计算量下最强版本（已经基本确定了该棋是先手必败的了）
(2.8 MB 200计算量胜2000计算量的Shenyqqq的友情项目https://huggingface.co/spaces/gumigumi/pmpmchess)
链接: https://pan.baidu.com/s/1CH9hwMLtLjuNl-3SizwOTQ?pwd=2333
提取码: 2333
也可以直接去weights文件夹下载最新的ONNX模型替换使用
```
如需根据计算资源修改参数，建议查看 self_play_worker.py 与 train_model.py 以及 C++ 代码  
六月初纯python写的旧项目连接：
```
https://github.com/liemark/popucom_chess
```
旧项目运行速度较慢，虽然也达到了人类平均水平，但花了整整好多天才训练出来

最后，比较可惜的是gui中单局对局（而非训练）的优化我是一点没做，所以gui中的搜索速度可以说是慢如蜗牛
# 泡姆棋（叭啵棋） (Match-n-POP)
>叭啵棋（Match-n-POP）是泡姆泡姆游戏内的官方名称，贴纸有提及

泡姆棋（叭啵棋）是鹰角网络的游戏泡姆泡姆（Popucom）的一种在 9×9 棋盘上进行的策略棋类游戏，结合了棋子放置、三消机制和区域涂色元素
## 游戏目标
在双方的步数用尽后，通过消除棋子和涂色地板，占据更多自己颜色的地板区域
## 游戏规则
### 1. 棋盘与棋子
棋盘大小为 9×9 格  
玩家分为黑方（红方）和白方（绿方），黑方先行  
每个玩家初始有 25 步（即 25 颗棋子）可供使用  
每个棋盘格有多种状态：  
存在黑子  
存在白子  
地板被涂成黑色  
地板被涂成白色  
地板未涂色（空地）
### 2. 落子规则
玩家轮流落子  
合法落子位置必须满足以下条件：  
该位置在棋盘范围内  
该位置当前没有任何棋子  
该位置的地板颜色与当前玩家的颜色匹配（例如，黑子只能下在黑地或空地上），或者该地板是未涂色的空地  
落子时，棋子会放置在该位置上，但不会立即改变脚下地板的颜色
### 3. 三消与涂色
当玩家落下一颗棋子后，游戏会检查是否连成了 3 颗或以上的同色棋子（包括刚刚落下的这颗棋子），检查方向包括：  
水平方向  
垂直方向  
两个对角线方向  
如果满足三消条件：  
消除： 连成一线的 n 颗同色棋子（n≥3）将被从棋盘上移除，如果同时满足多个三消条件（例如，一个子同时触发了横向和纵向三消），则所有满足条件的连子都会被消除  
涂色： 以刚刚落下的棋子为中心，沿着发生三消的所有行、列或斜线方向，将地板涂成当前玩家的颜色  
涂色会向外延伸，直到遇到对方的棋子为止（对方的棋子会阻碍涂色，无法改变对方的棋子下的地板颜色，己方的棋子不会阻碍涂色）  
涂色不会被已涂色的地板阻碍  
被消除棋子下方的地板也会被涂色
### 4. 终局判断
游戏在以下两种情况之一发生时结束：  
所有步数用尽： 当黑白双方的 25 步棋全部用尽后（此时最后一步棋一定是白方下的），游戏结束  
统计棋盘上黑方涂色地板的数量和白方涂色地板的数量  
涂色地板多的一方获胜  
如果双方涂色地板数量相同，则判为平局  
一方无法落子： 如果某一方在自己的回合开始时，发现棋盘上已经没有合法的落子位置，则该方立即判负，这通常意味着棋盘上大部分区域已被对方占据或填满
## 项目结构
本项目由多个 Python 文件和多个 C++ 文件组成：  
### run_pipeline.py
一个自动化脚本，负责循环运行 self_play_worker.py 自对弈生成棋谱、运行 train_model.py 生成 torch 的model.pth模型  
### run_pipeline_trt.py
一个自动化脚本，负责循环运行 self_play_worker_trt.py 自对弈生成棋谱、运行 train_model.py 生成 torch 的model.pth模型、运行 build_tensorrt_engine.py 生成 TensorRT 模型（与run_pipeline选其中一个用即可）  
### self_play_worker.py
负责生成自对弈数据，它使用当前训练好的 torch 神经网络 model.pth 和 MCTS 来自对弈，并记录游戏过程中的状态、MCTS 策略和最终结果，会同时进行大量对局以增加自对弈效率（建议根据自己的机器修改线程数）  
### self_play_worker_trt.py
负责生成自对弈数据，它使用当前训练好的 TensorRT 神经网络 model.plan 和 MCTS 来自对弈，并记录游戏过程中的状态、MCTS 策略和最终结果，会同时进行大量对局以增加自对弈效率（建议根据自己的机器修改线程数）  
### train_model.py
负责神经网络的训练，它加载 self_play_worker.py 生成的自对弈数据，并使用这些数据来更新神经网络的权重（建议根据自己的机器适当修改）
### build_tensorrt_engine.py
用 model.pth 生成临时的ONNX模型 model_temp.onnx 然后再生成 TensorRT 模型 model.plan  
### popucom_nn_interface.py 和 popucom_nn_model.py
定义泡姆棋的神经网络模型架构（基于残差块），包含策略头和价值头（通过在残差块中插入全局注意力模块进行了长程关系的改进，同时在注意力分数引入了相对坐标偏置以感知相对坐标）  
### popucom_chess_gui.py
游玩的ui界面，目前提供**人人/人机/机机**对弈功能，并加入了**对局树**~~虽然比较丑~~便于复盘分析，若使用onnx版本，运行速度会快不少，在运行onnx版本前，可以先运行**convert_to_onnx.py**以获得最新权重的onnx模型  
### popucom_chess_gui_onnx.py
游玩的ui界面的onnx引擎版本，可运行**convert_to_onnx.py**以获得model.pth对应的的onnx模型 model.onnx  
### convert_to_onnx.py
将 model.pth 转换为 model.onnx 以供 popucom_chess_gui_onnx.py 使用  
### arena_onnx.py
将 convert_to_onnx.py 生成的模型命名为 model_a.onnx 和 model_b.onnx 对打测试  
>暂时没做并行处理，懒得改了

### popucom_core.dll
由C++代码编译获得，负责 MCTS，如果爆内存建议手动修改
>置换表最大尺寸:puct.cpp中的DEFAULT_TT_MAX_SIZE
>每次储存的节点数：mcts_search.cpp中的INITIAL_NODE_STORE_CAPACITY

## 如何运行
确保您已安装 Python 3.x 和 PyTorch  
>TensorRT和ONNX可能还需要CUDA和CUDNN

各种软件包缺什么下什么  
对C++文件夹内所有文件编译获得 popucom_core.dll  
生成初始数据并训练:  
首次运行，由于没有 model.pth，self_play_worker.py 会使用随机权重模型  
运行训练：
```
python run_pipeline.py
```
如果有PyCharm，直接运行 run_pipeline.py 即可自对弈并训练模型
