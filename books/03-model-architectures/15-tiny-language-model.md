# M15　综合项目：从原始文本到训练、恢复、评估与生成

[本卷导航](README.md) · 前置：M01–M06、[后训练与多模态](14-training-modalities.md) · 下一章：[架构研究](16-architecture-study.md)

这一章把前面的知识连接成两个完整项目。项目 A 是 NumPy 字符上下文 MLP 语言模型，所有参数梯度都能手算和差分；项目 B 是手写 PyTorch Transformer，让自动微分负责复杂反向，但 attention、残差与数据目标仍由你明确实现。先完成小模型不是降低目标，而是建立可以逐项验算的训练基线。

本章使用本地人工语法数据，不下载权重，不调用付费 API。CPU 足以完成项目 A；项目 B 默认 CPU，是否使用个人显卡或服务器单卡由你显式选择。所有训练结果只说明模型学到了这个小数据分布，不代表通用语言能力。数据、模型、优化器、评估和生成分别有独立职责，不能用一段看起来像句子的输出替代整个验收。

## 第一课　先写任务契约：模型到底预测什么

项目 A 接收最近 C 个字符 ID，预测紧接着的一个字符。C 默认为三，输入例如 t、h、e，目标可能是空格。模型没有 attention，也没有无限历史；它只能根据固定上下文窗口建模。把这个模型叫 Transformer 会误导，因为它的参数、计算图与状态能力都不同。

语言模型学习的是条件分布，而不是每个上下文唯一确定的答案。同一个三字符片段在不同句子中可能接不同字符，模型应给这些结果分配概率。训练交叉熵衡量正确标签的负对数概率，生成时再根据分布选 ID。损失低与生成多样性之间的关系，需要结合数据歧义和采样策略理解。

本章数据包含猫、狗、跑、坐等英文短句的重复组合。它足够让模型学习字符模式，但训练与验证共享同一人工语法，不能用来证明对自然语言新领域的泛化。我们保留这个限制，因为它让你专注训练机制；扩展时会加入未见组合与独立文本验证，而不是偷偷把简单数据结果包装成大模型能力。

```bash
python -m labs.models.tiny_lm --steps 160
python -m labs.models.tiny_lm --steps 160 --save artifacts/models/tiny-lm.npz
```

第一条只训练与打印结果，第二条显式保存推理 checkpoint。CLI 限制训练步数，避免误输入导致无界运行。你应先在 CPU 上跑通，再决定是否需要更大实验；增加硬件不会修复标签错位、梯度遗漏或数据泄漏。

## 第二课　字符表、ID 与数据切分

字符表使用排序后的唯一字符集合，建立确定的字符到 ID 映射。排序不是语言学规则，而是可复现约定。embedding 的第 i 行与输出头第 i 列都绑定这个 ID 的字符含义，所以 checkpoint 必须同时保存 alphabet。只保存权重却重建不同顺序的词表，会让同一数字指向不同字符。

训练与验证应先按原始文本或独立文档划分，再各自构造窗口。若先对整段文本生成高度重叠窗口，再随机分样本，训练与验证可能共享几乎相同的字符片段，评估过于乐观。对于真实语料，还要检查重复文档、模板复制与近重复内容；本章只处理可见的连续窗口边界，并明确重复人工语法的局限。

长度 N 的文本、上下文 C，会产生 N 减 C 个样本。第 i 个输入是位置 i 到 i 加 C 减一，标签是 i 加 C。最后一个标签是原文本最后一个字符，最后一个输入窗口在它之前结束。这个计数可以手算，不应靠试着 reshape 到某个尺寸。

```python
import numpy as np
from labs.models.tiny_lm import TinyLanguageModel,dataset,CORPUS,train
from labs.models.core import cross_entropy,sample_token

small_text="ababa"
small_alphabet="ab"
small_x,small_y=dataset(small_text,small_alphabet,context=2)
np.testing.assert_array_equal(small_x,[[0,1],[1,0],[0,1]])
np.testing.assert_array_equal(small_y,[0,1,0])
assert len(small_x)==len(small_text)-2

alphabet="".join(sorted(set(CORPUS)))
boundary=int(len(CORPUS)*.8)
train_text=CORPUS[:boundary]
valid_text=CORPUS[boundary:]
x_train,y_train=dataset(train_text,alphabet,3)
x_valid,y_valid=dataset(valid_text,alphabet,3)
assert len(x_train)==len(train_text)-3
assert len(x_valid)==len(valid_text)-3
```

未知字符也需要明确策略。当前字符模型只接受 alphabet 中的字符，遇到未见字符会报错，没有偷偷映射成某个已有字符。若增加 unknown token，必须重新定义词表、训练数据与输出目标；若使用 byte tokenizer，可以覆盖任意 UTF-8 字节，但序列长度与语义单位改变。不要在生成时报错后临时删掉字符而不记录，因为那会改变用户输入。

## 第三课　batch 与随机数：一步训练不是一个 epoch

课程训练从窗口样本中有放回抽取六十四个索引，形成一个 batch。某个样本可能在同一 batch 或不同步骤重复出现，也可能一段时间没有被抽中。这样定义的是随机更新步数，不是每个 epoch 严格遍历所有样本。报告 steps 最准确，若估算“等效数据遍数”也应注明只是样本量比例。

随机数生成器控制 batch 抽样，模型初始化也有自己的种子。两者即使用相同数字作为 seed，仍应明确各自作用。保存训练恢复时，需要保存抽样 RNG 的内部状态，不能只保存初始 seed；恢复后从初始 seed 重新抽样会重复最早批次，而不是接着原来的数据流继续。

```python
batch_rng=np.random.default_rng(15)
indices=batch_rng.integers(0,len(x_train),size=64)
batch_x,batch_y=x_train[indices],y_train[indices]
assert batch_x.shape==(64,3)
assert batch_y.shape==(64,)
assert np.all((batch_x>=0)&(batch_x<len(alphabet)))
```

batch 大小改变会影响梯度噪声、吞吐和优化步数，但不应该改变参数 shape。bias 只有隐藏维或词表维，不能带 batch 轴；模型应能处理一、七、六十四等不同样本数。测试多个 batch shape 可以发现把训练批次大小写死进参数或 reshape 的问题。

有放回抽样还提醒你正确处理重复 embedding ID。同一个字符在多个样本和多个上下文位置出现时，其梯度贡献必须累加到同一参数行。数据中重复不是异常，而是语言建模的常态；反向实现不能假设索引唯一。

## 第四课　五组参数与完整前向

设词表 V、上下文 C、embedding 维 e、隐藏维 h。E 为 V 乘 e；查表后得到 batch、C、e，按上下文顺序拼成 batch、C e；W1 为 C e 乘 h，b1 为 h；tanh 后仍为 batch、h；W2 为 h 乘 V，b2 为 V。输出 logits 为 batch、V，每行对应一个下一字符分布。

拼接保留上下文位置。字符 a、b 与 b、a 即使使用同一 embedding 表，拼接后坐标顺序不同，W1 可以区分它们。若把上下文 embedding 直接求平均，就丢掉顺序，变成另一个模型。reshape 只是把已有位置轴展平，不应该重排 token 顺序。

```python
def independent_forward(model,ids):
    p=model.params
    looked_up=p["E"][ids]
    flat=looked_up.reshape(len(ids),model.context*model.embedding)
    preactivation=flat@p["W1"]+p["b1"]
    hidden=np.tanh(preactivation)
    logits=hidden@p["W2"]+p["b2"]
    return logits,{"ids":ids,"looked_up":looked_up,"flat":flat,
                   "preactivation":preactivation,"hidden":hidden}

model=TinyLanguageModel(len(alphabet),seed=15)
for batch_size in (1,7,64):
    ids=x_train[:batch_size]
    ours,saved=independent_forward(model,ids)
    reference,_=model.forward(ids)
    assert ours.shape==(batch_size,len(alphabet))
    np.testing.assert_allclose(ours,reference,atol=1e-12)
```

forward 保存 ids、展平 embedding 与 hidden，供反向使用。这种 cache 是当前训练调用的中间量，不是跨 token 的 KV cache。它随一次前向建立，反向后可以释放；生成时没有必要长期保存所有训练中间量。不同场景都叫 cache，读代码时要看生命周期而不是名字。

tanh 将每个预激活压到负一到一，导数为一减 hidden 平方。输入绝对值很大时接近饱和，导数很小。初始化让矩阵尺度适中，有助于避免一开始所有隐藏单元都饱和。后续模型可能用 GELU 或 SwiGLU，但本项目选择 tanh 是为了平滑且便于完整梯度检查。

## 第五课　一条样本的完整手算

构造词表三个字符、上下文二、embedding 一维、隐藏二维。E 的三行是零、一、二，W1 为二维单位矩阵，两个 bias 为零。输入 ID 为零、一，因此查表后拼接为零、一；预激活相同；隐藏为零与 tanh 一，记后者为 a。输出 W2 两行分别一、零、负一与零、一、负一，所以 logits 为零、a、负 a。

目标为 ID 一。令 Z 为一加 exp(a) 加 exp(-a)，概率为一除 Z、exp(a) 除 Z、exp(-a) 除 Z。损失为 log Z 减 a。logits 梯度为这三个概率分别减去零、一、零。这里 batch 只有一，所以没有额外除样本数；若 batch 有 N，平均交叉熵会把每行梯度除 N。

输出权重梯度是隐藏列向量与 logits 梯度的外积。第一行全零，第二行为 a 乘各 logits 梯度。隐藏梯度第一分量为第一类梯度减第三类梯度，第二为第二类梯度减第三类梯度。经过 tanh，第一分量乘一，第二乘一减 a 平方。W1 梯度第一行零，第二行等于这两个预激活梯度；embedding ID 零与一分别收到展平输入的两个梯度。

```python
hand_model=TinyLanguageModel(3,context=2,embedding=1,hidden=2,seed=0)
hand_model.params["E"][:]=np.array([[0.],[1.],[2.]])
hand_model.params["W1"][:]=np.eye(2)
hand_model.params["b1"][:]=0.
hand_model.params["W2"][:]=np.array([[1.,0.,-1.],[0.,1.,-1.]])
hand_model.params["b2"][:]=0.
hand_ids=np.array([[0,1]])
hand_labels=np.array([1])
a=np.tanh(1.)
z=1+np.exp(a)+np.exp(-a)
probability=np.array([1/z,np.exp(a)/z,np.exp(-a)/z])
g=probability-np.array([0.,1.,0.])
hidden_gradient=np.array([g[0]-g[2],g[1]-g[2]])
pre_gradient=hidden_gradient*np.array([1.,1-a*a])
hand_loss,hand_grads=hand_model.loss_and_grads(hand_ids,hand_labels)
np.testing.assert_allclose(hand_loss,np.log(z)-a)
np.testing.assert_allclose(hand_grads["W2"],np.outer([0.,a],g))
np.testing.assert_allclose(hand_grads["W1"],np.outer([0.,1.],pre_gradient))
np.testing.assert_allclose(hand_grads["b1"],pre_gradient)
np.testing.assert_allclose(hand_grads["b2"],g)
np.testing.assert_allclose(hand_grads["E"][:,0],[pre_gradient[0],pre_gradient[1],0.])
```

这条手算连接了查表、拼接、线性层、tanh、输出头、softmax 与交叉熵。你应能解释每个零梯度来自哪里：W2 第一行零是因为对应 hidden 为零，W1 第一行零是因为对应输入为零，未出现的 embedding 第三行没有路径。零梯度不总是 bug，要根据计算图判断。

## 第六课　完整反向与重复索引累加

下面独立写出所有参数梯度。先取得平均交叉熵对 logits 的梯度 G，计算输出头梯度与隐藏梯度，再乘 tanh 导数，得到 W1、b1 与展平 embedding 梯度。最后 reshape 回 batch、context、embedding，并按 ID 累加到 E。每个参数的 shape 与原参数完全相同，便于优化器逐项更新。

```python
def independent_loss_and_grads(model,ids,labels):
    logits,saved=independent_forward(model,ids)
    loss,g=cross_entropy(logits,labels)
    p=model.params
    hidden=saved["hidden"]
    flat=saved["flat"]
    gradients={"W2":hidden.T@g,"b2":g.sum(axis=0)}
    dpre=(g@p["W2"].T)*(1-hidden*hidden)
    gradients["W1"]=flat.T@dpre
    gradients["b1"]=dpre.sum(axis=0)
    dlookup=(dpre@p["W1"].T).reshape(len(ids),model.context,model.embedding)
    gradients["E"]=np.zeros_like(p["E"])
    np.add.at(gradients["E"],ids,dlookup)
    return loss,gradients

check_model=TinyLanguageModel(4,context=2,embedding=2,hidden=3,seed=16)
check_ids=np.array([[1,1],[1,2],[3,1]])
check_labels=np.array([2,3,0])
ours_loss,ours_grads=independent_loss_and_grads(check_model,check_ids,check_labels)
ref_loss,ref_grads=check_model.loss_and_grads(check_ids,check_labels)
np.testing.assert_allclose(ours_loss,ref_loss)
for name in check_model.params:
    np.testing.assert_allclose(ours_grads[name],ref_grads[name],atol=1e-12)
    numerical=np.zeros_like(check_model.params[name])
    parameter=check_model.params[name]
    for index in np.ndindex(parameter.shape):
        old=parameter[index]
        parameter[index]=old+1e-6
        plus=independent_loss_and_grads(check_model,check_ids,check_labels)[0]
        parameter[index]=old-1e-6
        minus=independent_loss_and_grads(check_model,check_ids,check_labels)[0]
        parameter[index]=old
        numerical[index]=(plus-minus)/2e-6
    np.testing.assert_allclose(numerical,ours_grads[name],atol=2e-7,rtol=2e-7)
```

这里检查了所有参数元素，不只每类抽两个坐标。模型很小，完整差分成本可接受，也能覆盖重复 ID 一的多次贡献。对大模型不能逐元素差分，但应保留这种小配置测试，再使用方向导数或抽样坐标检查。测试规模小不意味着证据浅，关键是它覆盖了完整计算图。

为什么不能写 grads_E[ids] 加等于 dlookup？NumPy 的高级索引会创建中间结果，重复索引的写回不一定按每次出现累加。np.add.at 明确执行散射累加，保证同一字符的所有贡献进入同一行。你也可以写显式双循环作为慢参考，先验证再优化。

```python
scatter_expected=np.zeros((4,2))
scatter_ids=np.array([[1,1],[2,1]])
scatter_values=np.arange(8,dtype=float).reshape(2,2,2)
for row in range(2):
    for position in range(2):
        scatter_expected[scatter_ids[row,position]]+=scatter_values[row,position]
scatter_actual=np.zeros((4,2))
np.add.at(scatter_actual,scatter_ids,scatter_values)
np.testing.assert_array_equal(scatter_actual,scatter_expected)
```

## 第七课　优化器、裁剪与实际训练日志

课程 update 使用 SGD：参数减去学习率乘梯度。更新之前计算所有参数梯度的全局欧氏范数，若超过阈值，就给所有梯度乘同一个系数。全局裁剪保留整体梯度方向，只限制长度；逐参数分别裁剪会改变不同参数组的相对比例，是另一种算法。报告应说明阈值与裁剪位置。

手算两组梯度分别三与四，全局范数五。阈值二点五时，统一缩放二分之一，得到一点五与二，范数二点五。若分别把每个分量夹到二点五，则得到二点五与二点五，方向和范数都不同。这个简单例子足以区分范数裁剪与逐元素裁剪。

```python
gradient_vector=np.array([3.,4.])
norm=np.linalg.norm(gradient_vector)
scale=min(1.,2.5/norm)
np.testing.assert_allclose(gradient_vector*scale,[1.5,2.])
np.testing.assert_allclose(np.linalg.norm(gradient_vector*scale),2.5)
```

训练日志至少记录步数、当前 batch 损失、固定训练子集损失、验证损失、梯度范数和是否裁剪。当前 batch 损失有抽样噪声，不必每一步下降；固定集合曲线更便于比较。验证时不更新参数，也不使用验证标签调 batch 梯度。若频繁依据验证结果选超参数，最终还需要独立测试集评估。

下面实际执行仓库训练函数。验收要求这个固定小实验的训练交叉熵明显下降，所有指标有限。阈值不是对任意语言模型的普遍保证，而是本地合成任务的回归条件。生成文本随后单独检查，不能用生成结果替代连续损失证据。

```python
trained_model,trained_alphabet,metrics=train(steps=160,seed=11)
assert metrics["final_train_ce"]<metrics["initial_train_ce"]-.5
assert np.isfinite([metrics["initial_train_ce"],metrics["final_train_ce"],
                   metrics["validation_ce"]]).all()
assert metrics["units"]=="nats/token"
```

学习率过大可能让损失振荡或出现非有限值，过小可能在有限步内几乎不动。先确认梯度正确和参数确实更新，再调整学习率。梯度裁剪可以控制异常步幅，但不能修复错位标签或错误反向；如果每一步都被极强裁剪，应调查尺度与初始化，而不是只把阈值继续降低。

## 第八课　评估：交叉熵、困惑度与数据边界

交叉熵使用自然对数，单位是每个目标 token 的 nats。困惑度是平均交叉熵的指数，直观上对应模型在这个分布上的有效不确定性尺度。若每个位置都对 V 个字符均匀预测，交叉熵为 log V，困惑度为 V。它不是“错误字符数量”，也不是准确率的简单倒数。

不同 tokenizer 的每 token 困惑度不能直接比较，因为 token 单位不同。字符模型与字节模型、BPE 模型的平均长度和词表都不同。即使使用相同 tokenizer，也要保持评估数据与 mask 相同。一个在重复人工语法上困惑度很低的模型，不一定能处理真实长文。

```python
def evaluate_ce(model,x,y,batch_size=128):
    total_loss=0.
    total_count=0
    for start in range(0,len(x),batch_size):
        stop=min(start+batch_size,len(x))
        logits,_=model.forward(x[start:stop])
        loss,_=cross_entropy(logits,y[start:stop])
        total_loss+=loss*(stop-start)
        total_count+=stop-start
    mean=total_loss/total_count
    return {"ce":float(mean),"perplexity":float(np.exp(mean)),"count":total_count}

validation=evaluate_ce(trained_model,x_valid,y_valid)
np.testing.assert_allclose(validation["ce"],metrics["validation_ce"],atol=1e-12)
uniform_ce=np.log(len(alphabet))
np.testing.assert_allclose(np.exp(uniform_ce),len(alphabet))
```

按 batch 评估时应按样本数加权平均。最后一个 batch 可能较短，如果把每批平均损失再简单平均，就给短批过大权重。代码先乘该批样本数累加总损失，再除总数。对 Transformer 的多位置目标，同样要按有效目标 token 数加权，padding 与忽略标签不计入分母。

训练文本和验证文本虽然在切分位置上隔离，本项目仍重复同一语法，因此验证衡量的是同分布模式学习。更强的扩展应按句子组合或模板划分，让某些主体动作组合只出现在验证；再加入独立文本，观察分布变化。不要为了让曲线好看把验证集换成训练样本，应该用清楚的任务范围解释结果。

## 第九课　推理 checkpoint：权重、配置与词表必须一起保存

仓库 save 把五组参数、模型配置与 alphabet 保存到 npz，load 禁止 pickle 并检查参数 shape。恢复后的同一输入 logits 应与保存前逐元素一致。生成文本还依赖采样种子，因此先比较确定的 logits，再比较固定种子的生成，能分离模型恢复与随机策略。

```python
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as temporary:
    checkpoint=Path(temporary)/"inference.npz"
    trained_model.save(checkpoint,trained_alphabet)
    restored_model,restored_alphabet=TinyLanguageModel.load(checkpoint)
    assert restored_alphabet==trained_alphabet
    before,_=trained_model.forward(x_valid[:17])
    after,_=restored_model.forward(x_valid[:17])
    np.testing.assert_array_equal(before,after)
    assert trained_model.generate("the",trained_alphabet,20,seed=3)==(
        restored_model.generate("the",restored_alphabet,20,seed=3))
```

这份基础 checkpoint 用于推理恢复，不包含训练抽样 RNG 与当前步数。SGD 无动量时优化器没有额外矩状态，但数据流仍需要恢复；换成 Adam 后还需要一阶矩、二阶矩与步计数。保存一个 state_dict 或 npz 并不自动意味着精确断点续训，必须明确保存范围。

词表顺序是权重语义的一部分。即使两个 alphabet 包含相同字符，顺序不同也不能直接互换。若确实要迁移，需要同时重排 E 的行、W2 的列与 b2 的元素，并处理新增或删除字符。仅把字符表字符串改掉，会让模型数值函数对应到错误文字。

## 第十课　完整断点续训：保存随机数与更新边界

下面实现一个完整的简单 SGD 训练运行器。它保存模型参数、结构、alphabet、训练步、抽样 RNG 状态、学习率、裁剪阈值、batch 大小和数据指纹。恢复时检查数据指纹，避免把旧训练状态接到不同语料却仍声称完全复现。代码使用临时目录测试，不污染仓库。

```python
import json
import hashlib

class TrainingRun:
    def __init__(self,alphabet,seed=11,lr=.3,clip_norm=5.,batch_size=64):
        self.alphabet=alphabet
        self.model=TinyLanguageModel(len(alphabet),seed=seed)
        self.rng=np.random.default_rng(seed+1)
        self.step=0
        self.lr=lr
        self.clip_norm=clip_norm
        self.batch_size=batch_size
    def advance(self,x,y,steps):
        losses=[]
        for _ in range(steps):
            idx=self.rng.integers(0,len(x),size=self.batch_size)
            loss,grads=self.model.loss_and_grads(x[idx],y[idx])
            self.model.update(grads,self.lr,self.clip_norm)
            self.step+=1
            losses.append(loss)
        return losses
    def save(self,path,data_fingerprint):
        config=[self.model.vocab,self.model.context,
                self.model.embedding,self.model.hidden]
        metadata={"alphabet":self.alphabet,"config":config,"step":self.step,
                  "rng":self.rng.bit_generator.state,"lr":self.lr,
                  "clip_norm":self.clip_norm,"batch_size":self.batch_size,
                  "data_fingerprint":data_fingerprint}
        np.savez(path,metadata=np.array(json.dumps(metadata)),
                 **{f"parameter_{k}":v for k,v in self.model.params.items()})
    @classmethod
    def load(cls,path,expected_fingerprint):
        with np.load(path,allow_pickle=False) as archive:
            metadata=json.loads(str(archive["metadata"]))
            if metadata["data_fingerprint"]!=expected_fingerprint:
                raise ValueError("training data fingerprint mismatch")
            run=cls(metadata["alphabet"],lr=metadata["lr"],
                    clip_norm=metadata["clip_norm"],batch_size=metadata["batch_size"])
            run.model=TinyLanguageModel(*metadata["config"])
            for name in run.model.params:
                value=archive[f"parameter_{name}"]
                if value.shape!=run.model.params[name].shape:
                    raise ValueError("parameter shape mismatch")
                run.model.params[name]=value.copy()
            run.rng.bit_generator.state=metadata["rng"]
            run.step=metadata["step"]
            return run

fingerprint=hashlib.sha256(train_text.encode("utf-8")).hexdigest()
continuous=TrainingRun(alphabet,seed=23)
continuous_losses=continuous.advance(x_train,y_train,80)
interrupted=TrainingRun(alphabet,seed=23)
first_losses=interrupted.advance(x_train,y_train,35)
with tempfile.TemporaryDirectory() as temporary:
    resume_path=Path(temporary)/"training.npz"
    interrupted.save(resume_path,fingerprint)
    resumed=TrainingRun.load(resume_path,fingerprint)
    second_losses=resumed.advance(x_train,y_train,45)
assert resumed.step==continuous.step==80
np.testing.assert_array_equal(first_losses+second_losses,continuous_losses)
for name in continuous.model.params:
    np.testing.assert_array_equal(resumed.model.params[name],continuous.model.params[name])
```

保存发生在某一步参数更新完成之后，恢复后下一次操作直接抽取下一批。若恢复代码为了打印示例先从同一个训练 RNG 抽样，就会消耗随机数，后续批次不同。评估与生成最好使用独立 RNG，或者完整保存它们各自状态，避免日志功能改变训练轨迹。

这个运行器没有学习率调度和动量，所以保存内容足以复现当前算法。加入 scheduler、Adam、混合精度 scaler 或数据增强后，需要把新增状态一并纳入。所谓完整 checkpoint 是相对于具体训练系统定义的，不是固定只有某几个文件字段。

## 第十一课　生成：模型分布与选择策略分开

生成先把 prompt 转为 ID，取最后 C 个字符作为输入，得到下一字符 logits，按策略选一个 ID，追加到历史，再重复。项目 A 每一步只看固定窗口，没有 KV cache；超出窗口的早期字符不会直接影响当前预测。这个限制由模型结构决定，不是生成循环写得不够长。

贪心选择最大 logit，结果确定但容易进入高概率循环；温度采样先缩放 logits，再随机选择；top-k 或 top-p 限制候选集合。它们改变从同一分布产生文本的方式，不会修复错误的模型训练。调试时先用贪心和固定种子采样，确认输出可复现，再研究多样性。

```python
def generate_ids(model,prompt_ids,new_tokens,seed=19,temperature=.8,top_p=.95):
    history=list(prompt_ids)
    if len(history)<model.context:
        raise ValueError("prompt shorter than context")
    generator=np.random.default_rng(seed)
    for _ in range(new_tokens):
        context=np.array([history[-model.context:]],dtype=int)
        logits,_=model.forward(context)
        next_id=sample_token(logits[0],generator,temperature=temperature,top_p=top_p)
        history.append(int(next_id))
    return history

stoi={character:index for index,character in enumerate(trained_alphabet)}
prompt_ids=[stoi[c] for c in "the"]
generated=generate_ids(trained_model,prompt_ids,40,seed=19)
generated_text="".join(trained_alphabet[index] for index in generated)
assert generated_text.startswith("the")
assert len(generated_text)==43
assert generated==generate_ids(trained_model,prompt_ids,40,seed=19)
```

不要把生成长度混成字符数与 token 数。在本项目字符 tokenizer 中二者恰好一致；换成字节或 BPE 后，生成四十个 token 不一定是四十个可见字符。解码还可能遇到多字节字符分段，第一章已经讨论过。模型输出 ID，用户看到文本，中间 tokenizer 契约必须完整。

一个短窗口字符模型可能生成重复句式，这是任务与容量共同导致的行为。应记录不同 prompt、不同种子和策略，而不只展示最漂亮的一条。若模型总输出同一个字符，先检查数据频率、输出 bias、loss、标签与梯度，再考虑采样温度；采样不能凭空创造未学到的条件关系。

## 第十二课　项目 B：完整手写 Transformer 与自动微分

项目 A 让你完全掌握梯度，项目 B 把序列算子换成真正的 causal Transformer。下面内联仓库的完整模型构建函数，不用 nn.Transformer 或 MultiheadAttention 隐藏主体。它包含 token embedding、可学习位置 embedding、多个 PreNorm block、手写 QKV、多头重排、因果 mask、softmax 读取、输出投影、GELU FFN、最终 norm 与词表头。

输入 ID 为 batch、time，embedding 后是 batch、time、dim。QKV 一次线性投影到三倍 dim，再沿特征轴切成三份。每份先 reshape 为 batch、time、heads、head_dim，再 transpose 成 batch、heads、time、head_dim。分数是 batch、heads、query_time、key_time，softmax 沿最后键轴。输出再逆向合并头，contiguous 保证转置后的布局能正确 view。

每个 block 先对当前 x 做 LayerNorm，attention 增量加回 x；再对更新后的 x 做第二次 LayerNorm，FFN 增量加回。位置 embedding 按当前窗口的位置零到 T 减一加入。这个模型使用学习绝对位置，不是前章 RoPE/GQA/SwiGLU 配置，参数与缓存规则也不同。它是另一个明确的可训练基线，不能直接加载前章随机模型数组当作兼容权重。

```python
import math
def build_model(torch,vocab,dim=32,heads=4,layers=2,context=24):
    nn=torch.nn
    class Attention(nn.Module):
        def __init__(self):
            super().__init__(); self.qkv=nn.Linear(dim,3*dim,bias=False); self.output=nn.Linear(dim,dim,bias=False)
            self.register_buffer('mask',torch.tril(torch.ones(context,context,dtype=torch.bool)))
        def forward(self,x):
            b,t,d=x.shape; q,k,v=self.qkv(x).chunk(3,dim=-1)
            def heads_view(z): return z.view(b,t,heads,d//heads).transpose(1,2)
            q,k,v=map(heads_view,(q,k,v)); scores=(q@k.transpose(-1,-2))/math.sqrt(d//heads)
            scores=scores.masked_fill(~self.mask[:t,:t],float('-inf'))
            out=torch.softmax(scores,dim=-1)@v
            return self.output(out.transpose(1,2).contiguous().view(b,t,d))
    class Block(nn.Module):
        def __init__(self):
            super().__init__(); self.n1=nn.LayerNorm(dim); self.n2=nn.LayerNorm(dim); self.attn=Attention()
            self.ff=nn.Sequential(nn.Linear(dim,4*dim),nn.GELU(),nn.Linear(4*dim,dim))
        def forward(self,x):
            x=x+self.attn(self.n1(x)); return x+self.ff(self.n2(x))
    class Model(nn.Module):
        def __init__(self):
            super().__init__(); self.token=nn.Embedding(vocab,dim); self.position=nn.Embedding(context,dim)
            self.blocks=nn.ModuleList([Block() for _ in range(layers)]); self.norm=nn.LayerNorm(dim); self.head=nn.Linear(dim,vocab)
        def forward(self,ids):
            x=self.token(ids)+self.position(torch.arange(ids.shape[1],device=ids.device))
            for block in self.blocks: x=block(x)
            return self.head(self.norm(x))
    return Model()

```

模型定义之后，完整训练循环如下。代码显式检查可选 PyTorch；环境缺少依赖时打印 SKIP，不伪装训练结果。安装与硬件配置按基础卷的环境课程处理，本章不提供未经当前环境核对的 CUDA 安装命令。默认函数只用 CPU，步数有明确参数，不会自动占用服务器多卡。

```python
def run_torch_project(steps=40):
    import importlib.util
    if importlib.util.find_spec("torch") is None:
        print("SKIP: optional PyTorch unavailable; Transformer training not executed.")
        return {"status":"skipped","reason":"torch unavailable"}
    import torch
    torch.manual_seed(15)
    torch.set_num_threads(2)
    torch_alphabet="".join(sorted(set(CORPUS)))
    mapping={c:i for i,c in enumerate(torch_alphabet)}
    data=torch.tensor([mapping[c] for c in CORPUS],dtype=torch.long)
    split=int(len(data)*.8)
    training,validation=data[:split],data[split:]
    context=12
    network=build_model(torch,len(torch_alphabet),dim=16,heads=4,layers=2,context=context)
    optimizer=torch.optim.AdamW(network.parameters(),lr=.003)
    generator=torch.Generator().manual_seed(150)
    def make_batch(stream,batch_size=8):
        starts=torch.randint(len(stream)-context,(batch_size,),generator=generator)
        inputs=torch.stack([stream[int(s):int(s)+context] for s in starts])
        labels=torch.stack([stream[int(s)+1:int(s)+context+1] for s in starts])
        return inputs,labels
    def loss_for(inputs,labels):
        logits=network(inputs)
        return torch.nn.functional.cross_entropy(
            logits.reshape(-1,len(torch_alphabet)),labels.reshape(-1))
    fixed_x,fixed_y=make_batch(validation)
    network.eval()
    with torch.no_grad():
        initial=float(loss_for(fixed_x,fixed_y))
    network.train()
    for _ in range(steps):
        inputs,labels=make_batch(training)
        loss=loss_for(inputs,labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(),1.)
        optimizer.step()
    network.eval()
    with torch.no_grad():
        final=float(loss_for(fixed_x,fixed_y))
        changed=fixed_x.clone()
        changed[:,-1]=(changed[:,-1]+1)%len(torch_alphabet)
        torch.testing.assert_close(network(fixed_x)[:,:-1],
                                   network(changed)[:,:-1],atol=1e-6,rtol=1e-5)
        generated=torch.tensor([[mapping[c] for c in "the"]])
        for _ in range(24):
            logits=network(generated[:,-context:])[:,-1,:]
            token=logits.argmax(dim=-1,keepdim=True)
            generated=torch.cat([generated,token],dim=1)
    assert np.isfinite([initial,final]).all()
    return {"status":"executed","device":"cpu","steps":steps,
            "initial_validation_ce":initial,"final_validation_ce":final,
            "text":"".join(torch_alphabet[i] for i in generated[0].tolist())}

optional_torch_result=run_torch_project()
```

标签与输入错开一个位置，每条长度 context 的输入对应同样长度的下一 token 目标。模型一次输出所有位置 logits，训练把 batch 与 time 展平后计算交叉熵。因果 mask 保证每个位置看不到其目标所在的未来输入。teacher forcing 让已知序列并行参与训练，不等于允许未来信息泄漏。

优化循环顺序是前向损失、清空旧梯度、反向、裁剪、更新。若忘记清空梯度，默认会跨步骤累加，等价于另一种未定义的训练过程。eval 与 no_grad 分别控制模块行为与计算图记录，二者职责不同；即使这个小模型没有 dropout，也应保持清晰的训练评估边界，方便后续扩展。

生成时每步裁取最近 context 个 token，位置从当前窗口零开始重新编号。这是该 toy 的明确约定，不是无限长 KV cache。超过训练窗口后的行为需要谨慎解释。若改成 RoPE 与缓存生成，应按 M05/M06 重新设计位置和状态，并增加 full/step 等价测试，不能只把生成循环里的切片删除。

## 第十三课　十二组练习与完整参考答案

### 练习一：独立构造窗口与标签

文本 abcde，上下文三，应有两个样本：abc 预测 d，bcd 预测 e。若出现 cde 预测不存在的位置，说明边界多走一步；若 abc 预测 c，说明标签没有右移。完整答案使用明确词表并检查每个 ID，不只检查数组长度。

```python
window_x,window_y=dataset("abcde","abcde",3)
np.testing.assert_array_equal(window_x,[[0,1,2],[1,2,3]])
np.testing.assert_array_equal(window_y,[3,4])
```

### 练习二：参数账本

默认词表十六、上下文三、embedding 八、隐藏二十四。E 有一百二十八个参数，W1 为二十四乘二十四共五百七十六，b1 二十四，W2 二十四乘十六共三百八十四，b2 十六，总一千一百二十八。上下文从三变五，只有 W1 输入宽度增加十六，参数增加十六乘二十四等于三百八十四。

```python
assert len(alphabet)==16
parameter_count=sum(value.size for value in model.params.values())
assert parameter_count==1128
longer=TinyLanguageModel(len(alphabet),context=5)
assert sum(value.size for value in longer.params.values())-parameter_count==384
```

参数量增加不自动保证任务提升。更长上下文提供更多输入位置，也增加样本稀疏性与优化负担；如果任务只需短局部信息，收益可能有限。比较时应保持数据与训练预算明确，而不是只报告更大模型训练损失更低。

### 练习三：重复 embedding 的完整梯度

第六课已经给出显式循环与 np.add.at 的参考。你还应构造同一个 ID 在同一窗口出现两次的例子，验证两次位置梯度都进入同一行。不要把重复样本去重来躲避问题，因为语言数据天然重复，去重会改变训练分布而不是修复反向。

### 练习四：平均损失的 batch 复制不变量

把同一个 batch 完整复制两遍，平均交叉熵与平均参数梯度应保持相同。这个测试能发现遗漏除 batch 数或重复累计尺度错误。若损失定义为求和，则结果应翻倍，必须按实际归约约定判断。

```python
base_loss,base_grads=check_model.loss_and_grads(check_ids,check_labels)
double_loss,double_grads=check_model.loss_and_grads(
    np.concatenate([check_ids,check_ids]),np.concatenate([check_labels,check_labels]))
np.testing.assert_allclose(base_loss,double_loss)
for name in base_grads:
    np.testing.assert_allclose(base_grads[name],double_grads[name],atol=1e-12)
```

### 练习五：一次更新确实改变参数

在固定 batch 上保存参数副本，执行一次梯度更新，确认至少一组参数发生变化，再检查更新方向与学习率公式。不能只看到 loss 变量改变就认为训练生效，因为不同 batch 自身就会产生不同损失。下面关闭裁剪影响，直接验证 SGD。

```python
update_model=TinyLanguageModel(4,context=2,embedding=2,hidden=3,seed=17)
_,update_grads=update_model.loss_and_grads(check_ids,check_labels)
before_update={name:value.copy() for name,value in update_model.params.items()}
update_model.update(update_grads,lr=.01,clip_norm=1e9)
for name in before_update:
    np.testing.assert_allclose(update_model.params[name],
        before_update[name]-.01*update_grads[name],atol=1e-12)
assert any(not np.array_equal(before_update[name],update_model.params[name])
           for name in before_update)
```

### 练习六：推理保存与训练恢复分别验收

推理恢复要求相同输入 logits 一致、词表与配置一致；精确续训还要求后续 batch、损失序列和最终参数一致。第九课与第十课分别给出完整答案。不能只测试 load 不报错，因为错误词表顺序或缺失 RNG 不一定触发 shape 异常。

### 练习七：完整早停与最佳参数恢复

每隔固定步数评估验证损失，若改善则复制全部参数作为最佳快照；连续若干次没有改善则停止。返回前恢复最佳参数，而不是最后参数。完整实现如下，另用学习率零制造确定的无改善情形，验证 patience 逻辑真的执行。

```python
def train_with_early_stop(max_steps=100,interval=5,patience=3,lr=.3):
    run=TrainingRun(alphabet,seed=31,lr=lr)
    initial=evaluate_ce(run.model,x_valid,y_valid)["ce"]
    best_loss=initial
    best_step=0
    best={name:value.copy() for name,value in run.model.params.items()}
    stale=0
    history=[(0,initial)]
    for _ in range(max_steps):
        run.advance(x_train,y_train,1)
        if run.step%interval:
            continue
        value=evaluate_ce(run.model,x_valid,y_valid)["ce"]
        history.append((run.step,value))
        if value<best_loss:
            best_loss=value
            best_step=run.step
            best={name:parameter.copy() for name,parameter in run.model.params.items()}
            stale=0
        else:
            stale+=1
        if stale>=patience:
            break
    for name in run.model.params:
        run.model.params[name]=best[name].copy()
    restored=evaluate_ce(run.model,x_valid,y_valid)["ce"]
    np.testing.assert_allclose(restored,best_loss,atol=1e-12)
    return run,{"best_step":best_step,"best_loss":best_loss,
                "stopped_step":run.step,"history":history}

early_run,early_report=train_with_early_stop()
flat_run,flat_report=train_with_early_stop(lr=0.)
assert flat_report["stopped_step"]==15
assert flat_report["best_step"]==0
assert early_report["best_loss"]<=early_report["history"][0][1]
```

最佳快照必须复制数组。若写 best 等于 model.params，或者只复制字典不复制数组，后续原地更新会改写所谓最佳参数。这个所有权问题与前章深度来源快照相同。保存模型、记录中间量和早停都需要明确引用与副本。

### 练习八：生成的可复现性与随机性

相同模型、prompt、采样策略和 seed 应生成相同 ID 序列；不同 seed 可以不同，但不保证一定不同，因为分布可能极尖或候选只剩一个。测试不能断言任何两个种子都必然产生不同文本。更合理的是验证固定种子可复现，并统计多次采样分布是否合理。

### 练习九：故意交换词表的风险

把 alphabet 前两个字符交换，而权重不变，数值 logits 完全相同，但它们对应的文字含义变了。完整答案应指出这不是模型计算误差，而是 tokenizer 与权重契约被破坏。若要正确重排，需要对输入 embedding 行和输出头列、bias 同步应用相同语义排列。

```python
permutation=np.arange(len(trained_alphabet))
permutation[[0,1]]=permutation[[1,0]]
permuted_alphabet="".join(trained_alphabet[i] for i in permutation)
assert set(permuted_alphabet)==set(trained_alphabet)
assert permuted_alphabet!=trained_alphabet
assert trained_alphabet[0]!=permuted_alphabet[0]
```

### 练习十：训练下降不能替代因果性

Transformer 忘记 causal mask 后，训练可能更容易预测标签，因为当前位置能直接看到后面的目标 token。这种损失下降是泄漏，不是学习能力提高。项目 B 的未来 token 修改测试是必要验收：只改最后输入，更早 logits 应不变。若环境没有 Torch，应明确未执行，不用 NumPy MLP 测试冒充 Transformer 验证。

### 练习十一：评估平均的正确权重

三批有效样本数二、二、一，平均损失分别一、三、十。正确总体平均为二加六加十除五，等于三点六；简单平均三批得到十四除三，错误地给最后一个样本与前两整批相同权重。这个手算适用于最后短 batch 和变长 token mask。

```python
counts=np.array([2,2,1])
losses=np.array([1.,3.,10.])
np.testing.assert_allclose(np.dot(counts,losses)/counts.sum(),3.6)
assert not np.isclose(losses.mean(),3.6)
```

### 练习十二：完整工程报告

报告包含数据来源与切分、tokenizer、模型配置、参数量、优化器与裁剪、训练步数、固定评估指标、保存恢复测试、生成策略和实际设备。每个结果标明来自哪次运行。没有运行 GPU 就不填写 GPU 吞吐，没有真实语料就不宣称自然语言泛化。完整项目的价值在于链条可复现，而不是堆砌一个很大的模型名称。

## 第十四课　故障工作坊：按证据定位训练问题

### 十四点一　loss 不降时先检查信息流

先打印三条输入窗口与标签，用字符还原后人工确认下一字符位置。再在微型配置上跑全部梯度差分，确保每组参数导数正确。随后检查一次更新前后参数是否变化，学习率是否为零，梯度是否被过度裁剪。最后才考虑容量、初始化或训练步数。这个顺序能避免用更大模型掩盖基础错误。

如果 loss 始终接近 log V，可能模型仍接近均匀分布，但原因不唯一。输出头未更新、隐藏全零、标签随机、学习率过小或数据目标本身高度不确定，都可能产生类似数字。应结合 logits 范围、梯度范数和小集合过拟合测试判断，不能仅凭一个曲线形状猜结论。

固定少量样本反复训练是有效诊断。若模型连十个确定样本都无法明显降低损失，优先怀疑实现或优化；若能过拟合但验证不好，再研究泛化与数据分布。过拟合小集合是检查工具，不是最终质量证明。报告应把这个诊断实验与正式验证分开。

### 十四点二　训练低、验证高：先分清过拟合与分布变化

训练损失下降而验证损失上升，可能是过拟合，也可能是验证数据分布不同、tokenizer 覆盖不同或预处理不一致。先检查两者是否使用同一字符映射、同一窗口长度与同一损失定义，再比较字符频率和上下文模式。若验证含未知字符，当前模型应明确拒绝，而不是静默删除后评估。

重复人工语法的验证很容易显得漂亮，因为训练已经见过几乎全部局部模式。扩展时可以把某些完整句型保留到验证，但要注意固定三字符模型可能无法区分长距离结构；任务难度应与模型上下文能力匹配。若目标需要五十字符前的信息，C 等于三的 MLP 理论上就看不到，不能靠更多训练步解决。

### 十四点三　非有限值与饱和

若参数或 logits 出现 NaN，先定位第一处非有限中间量。检查学习率、梯度范数、交叉熵稳定实现和输入数值。不要在最终 loss 上简单把 NaN 替换成零继续训练，这会掩盖计算图已经失效。保存最近正常 checkpoint 与触发 batch，才能重现实验并修复。

tanh 饱和时，隐藏值接近正负一，导数接近零，前层梯度可能很小。统计预激活绝对值和 hidden 分布，可以判断是否初始化或更新把模型推入饱和。输出头仍可能学习，因此 loss 缓慢下降不表示所有层都正常工作。逐组梯度日志有助于发现某些参数长期几乎不更新。

### 十四点四　生成退化不能只调温度

模型反复输出空格或某个字符，可能来自数据频率、输出 bias、上下文不足、训练不充分或采样策略。先用固定上下文打印完整概率分布，看模型是否真的偏向一个字符；再检查这些上下文在训练数据中的标签分布。如果数据本来高度重复，模型重复并不一定是实现错误。

温度升高会增加随机性，但可能破坏已经学到的语法；降低温度可能增强重复。top-p 和 top-k 也会改变候选范围。应该把模型质量与解码策略分别实验，不把一个更随机的输出误认为模型更懂语言。生成展示应保留多个种子与失败样例，避免只选最好看的结果。

### 十四点五　恢复后结果不同的排查顺序

如果恢复后同一输入 logits 就不同，检查权重、配置、词表与 dtype；如果 logits 相同但采样不同，检查 RNG 与采样参数；如果第一步训练一致、后续开始不同，检查数据抽样顺序、优化器状态、scheduler 和随机模块。分层定位比笼统地说“随机性导致”更有用。

跨硬件或不同矩阵库可能产生浮点差异，即使 seed 一样也未必逐位一致。本章精确续训测试在同一 CPU/NumPy 环境和相同执行顺序下成立。迁移设备时应改为合理容差与任务指标验证，并记录版本；不能把本地逐位复现承诺无条件扩展到所有平台。

## 第十五课　从小模型走向可维护的学习工程

### 十五点一　把文件职责分开

一个完整项目至少应区分数据准备、模型定义、损失与优化、训练入口、评估入口、生成入口和 checkpoint 格式。它们可以在小项目里共处一个文件，但概念职责要清晰。后续扩展时按职责拆分，比把所有内容塞进一个训练循环更容易测试与复用。

数据模块负责字符表、切分与窗口，不应偷偷更新模型；模型 forward 负责确定的 logits，不应在内部采样下一个字符；训练器负责梯度与更新，不应把验证集混进训练 batch；生成器负责策略与停止，不应改变权重。明确边界可以让你替换模型结构而保留数据与评估，也能让你更容易定位错误。

### 十五点二　实验配置与结果记录

每次实验保存结构配置、训练超参数、数据指纹、seed、代码版本与运行设备。结果至少包括初始与最终训练损失、验证损失、训练步数、参数量和耗时。若比较两个组件，先写清唯一改变的因素，其他条件保持一致或明确说明差异。

不要把一次随机种子的结果当成普遍结论。小模型可以用几个固定种子重复，报告均值与范围；如果某配置只在一个种子上好，应调查初始化敏感性。更长上下文、更宽 hidden 或不同激活的比较，还要说明参数与计算是否相同，否则无法把收益完全归因于某个机制。

### 十五点三　从 MLP 到 Transformer 的迁移顺序

先保留同一字符表和原始数据切分，把单目标窗口改成多位置右移目标；再实现 embedding 与位置；然后接一层手写 causal attention 与 FFN，验证未来不干扰；最后增加层数和训练。不要同时换 tokenizer、语料、优化器和模型结构，否则任何变化都难解释。

自动微分减少手写反向工作，但不会替你验证标签、mask、reshape 或状态。一个未来可见的 Transformer 可以有完美的自动梯度，却优化了错误任务。一个错误合并头的实现也能训练，只是训练另一个函数。前章的数值参考与结构不变量仍应保留。

### 十五点四　硬件安排应服务于实验问题

项目 A 全部在 CPU 上完成，方便逐元素差分与断点调试。项目 B 的小配置也可先用 CPU，确认完整流程后再在个人显卡测实际吞吐。需要更大 batch、长序列或多卡算法时，再使用服务器单卡或短时多卡实验，并设置步数上限和结束清理。硬件选择依据测量目标，而不是模型学习必须从八卡开始。

性能报告记录模型维度、batch、context、dtype、训练或推理模式、是否包含数据准备与保存。GPU 计时要处理异步执行，峰值显存也应说明统计范围。没有运行的设备不填写猜测数据；理论参数量和实际显存是不同证据。用户拥有强硬件是一种资源，课程应让你知道何时使用它能回答问题。

### 十五点五　第一版的完成门槛

你应能从一段文本构造正确目标，独立写出模型前向与全部 NumPy 梯度，实际训练并解释损失，保存恢复推理，精确续训，实施早停，生成多个可复现样例，并在手写 Transformer 中验证因果性。完成这些环节后，你拥有的是一个训练工程基线，而不只是复制过一段 loss 循环。

进一步扩展应在这个基线上选择明确任务：替换 RMSNorm、RoPE、SwiGLU 或 MoE，加入 KV cache，测试更长上下文，或者比较固定状态层。每个改动都配对应公式、shape、单元测试与训练对照。这样前面各章的机制才能真正进入同一个可维护系统，而不是停留在孤立 notebook 里。

### 十五点六　梯度累积的完整工程答案

显存不足时可以把一个大 batch 拆成几个 microbatch，分别前向反向，再累积梯度，最后只更新一次。要与大 batch 的平均损失等价，每个 microbatch 的平均梯度应按它的样本数占总数的比例加权。若最后一个 microbatch 较小，不能把所有批次简单平均。优化器更新次数也必须保持一次，否则变成不同训练过程。

下面使用三个样本拆成一加二，逐参数验证累积梯度与一次大 batch 一致。这个小测试可以迁移到 Transformer 的有效 token 数加权；有 padding 或忽略标签时，权重应按有效目标数，而不是只按 batch 行数。梯度裁剪应在完整累积之后执行，否则每个小批分别裁剪会改变方向与尺度。

```python
full_loss,full_grads=check_model.loss_and_grads(check_ids,check_labels)
accumulated={name:np.zeros_like(value) for name,value in check_model.params.items()}
weighted_loss=0.
for start,stop in ((0,1),(1,3)):
    loss,grads=check_model.loss_and_grads(check_ids[start:stop],check_labels[start:stop])
    fraction=(stop-start)/len(check_ids)
    weighted_loss+=fraction*loss
    for name in accumulated:
        accumulated[name]+=fraction*grads[name]
np.testing.assert_allclose(weighted_loss,full_loss,atol=1e-12)
for name in accumulated:
    np.testing.assert_allclose(accumulated[name],full_grads[name],atol=1e-12)
```

累积可以减少单次激活内存，但不会减少模型参数和优化器状态。它也可能改变随机模块的调用与数值归约顺序，因此在复杂模型中不一定逐位相同。对于本章确定的 NumPy MLP，等价关系清楚，适合作为你实现更大系统前的基准。

### 十五点七　怎样记录一次真正执行过的结果

本章这次 CPU 运行的固定一百六十步实验，初始训练交叉熵约二点七四五一，最终训练交叉熵约零点二三三二，验证交叉熵约零点二三四四，单位均为每字符 nats。数字来自本地人工重复语法、固定 seed 和当前实现，只用于说明训练闭环确实执行且损失下降。换版本、精度或随机种子后可能略有变化，不应把这些小数当作必须永久一致的能力指标。

同次验证执行了所有参数的中心差分、推理保存恢复、八十步连续训练与三十五步加四十五步恢复训练的逐位一致、早停最佳参数恢复和固定种子生成。可选 PyTorch 在本环境未安装，函数明确返回跳过，因此本章没有宣称 Transformer 训练或 GPU 性能已经运行。读者安装依赖后，应记录自己的实际结果，而不是复制本文 CPU 指标填表。

实验记录应把“通过什么断言”与“观察到什么数值”分开。断言表达契约，例如恢复前后 logits 一致；数值描述一次运行，例如验证损失。前者在符合条件的环境中应稳定成立，后者可以随数据和训练变化。把两者区分，会让报告更容易审核与复现。

### 十五点八　数据扩展的下一步不是直接换巨大语料

先把人工语法拆成独立句子，按句子组合划分训练与验证，确保不跨文档拼接窗口。若需要文档边界，可以加入明确的起止符并定义它们是否参与预测。直接把文件内容连成一条长字符串，可能让模型学习上一文档结尾接下一文档开头的伪模式；这可以是某种训练约定，但应明确而非偶然。

再选择规模可控、来源允许使用的文本，记录编码、清洗、去重与切分规则。不要先追求数据量，先检查 tokenizer 覆盖、样本长度分布和标签正确性。对公司资源与资料，只使用已经获准的学习数据；本项目本身不需要接触任何公司私有语料，公开或自造小数据足以完成机制训练。

扩展到 BPE 时，词表与模型输出维度都改变，不能继续沿用字符级困惑度进行直接比较。扩展到长上下文时，固定 MLP 的 W1 会随上下文线性增长，Transformer 的 attention 激活则可能随长度平方增长。不同结构的资源曲线不同，应先算账并小规模测量，再决定使用个人显卡还是服务器。

### 十五点九　项目答辩的五个问题

第一，给你一个输入窗口，能否不用运行代码写出每个张量的 shape，并指出标签位置？第二，给你一组重复 token ID，能否解释 embedding 梯度怎样累计？第三，恢复训练后下一批不同，能否定位 RNG、数据顺序或额外随机调用？第四，训练损失很好却生成退化，能否区分模型分布与采样策略？第五，把一个结构组件换掉后，能否说明 checkpoint 为什么可能不兼容？

这些问题不要求背诵所有 API，而要求你掌握数据与状态的因果链。能够回答并用代码验证，就具备继续深挖 Agent、框架、算子与新模型结构的训练基础。后续每次扩展都应保留这套小型完整基线，让新技术有一个可以比较、可以回退、可以解释的起点。

最后保留一份失败样例集：未知字符、过短 prompt、错误上下文宽度、负 token ID、超词表 ID、损坏的参数 shape 和不匹配数据指纹。它们应产生明确异常或按事先定义的策略处理，而不是静默截断或随机填补。正常训练跑通之后再执行这些边界检查，能确认工程接口对学习者友好，也能防止后续扩展把数据错误伪装成模型质量问题。

保存前后都应检查参数有限性，避免把已经损坏的训练状态当作最佳模型发布。
