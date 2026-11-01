# K10 算子进入框架：从反向图、梯度累加到自定义算子、调度与编译合同

写出正确GPU Kernel只是框架集成的一部分。框架还需要知道它接收什么、返回什么、是否修改输入、输出是否共享存储、如何传播梯度、没有真实数据时如何推断元信息，以及不同设备上选择哪种实现。本章从一个最小自动微分引擎开始，让这些职责不再只是名词，再把RMSNorm与自定义平方算子接入PyTorch的实际接口。

先修Python函数与类、K05反向公式、K09算子合同。多数概念与NumPy实验不需要GPU；PyTorch示例即使在CPU也能完成。当前交付环境没有PyTorch，所以框架代码未实际执行，NumPy参考与数学推导可独立验证。API细节以SOURCES中的官方文档和实际安装版本为准，不能把示例展示当作某版本的运行证据。

## 第一课：算子、Kernel、模块与计算图不是同一层

### 1. 数学函数与平台实现

数学算子定义输入到输出的关系，例如平方、矩阵乘法或RMSNorm。Kernel是在某个平台上执行该关系的一种实现。一个算子可以有CPU、CUDA、不同dtype与布局的多条Kernel路径；一个融合Kernel也可以实现若干数学步骤。

模块通常还管理参数与状态，例如线性层保存weight与bias。模块调用一个或多个算子，算子再经调度选择实现。自动微分图记录实际执行中的依赖与反向规则，并不等同于Python类层次。把这几层区分开，才能知道修改某处代码会影响数学、参数管理、调度还是执行性能。

### 2. 一个平方函数的四种责任

数学定义是y=x平方。前向实现可以用框架乘法、C++循环、CUDA或Triton。反向规则是输入梯度等于上游梯度乘二x。元信息规则是输出shape、dtype与device如何由输入决定。别名规则则说明输出是否新分配，是否与x共享storage。

即使前向数值正确，其他责任也可能错误。例如返回x本身虽然shape一致，却不是平方；返回x的某个视图可能数值经过原地修改后看似正确，却违反新输出合同；fake实现返回错误stride，编译时可能出问题；反向漏乘上游梯度，会在复杂图中失效。

### 3. 为什么框架不能只把它当黑盒函数

执行推理时，框架至少需要知道如何调用与管理内存；训练时需要梯度；编译时需要理解输入输出元信息与副作用；设备分派需要选择正确实现。黑盒可以作为边界存在，但边界必须提供足够合同，不能要求框架猜测内部行为。

自定义算子不是越多越好。若现有框架表达式已经清楚且能被编译器优化，优先保留组合表达式可能更简单。自定义边界适合外部实现、特定融合、稳定接口或框架无法直接理解的计算。选择应服务明确问题，而不是为了显得更底层。

## 第二课：链式法则与上游梯度

### 1. 为什么backward必须接收g

设y=x平方，损失L=三y，则dL/dy为三，dL/dx为三乘二x。backward接收的g就是后续图传来的dL/dy。只返回二x相当于假设g永远为一，仅在损失恰好等于y时正确。

向量输出也一样。若L是输出与固定权重g的内积，输入梯度是雅可比转置乘g，而不是简单列出每个输出对输入的局部导数。实际框架通常计算这种向量雅可比积，避免显式构造巨大雅可比矩阵。

### 2. 一个有分支的手算图

令a=x平方，b=三x，L=a+b。x为二时，a为四，b为六，L为十。平方路径贡献四，线性路径贡献三，所以x梯度为七。两个路径都连接同一个x，反向必须把贡献相加，不能后来的路径覆盖前一个梯度。

再令L=a+a，虽然图中两个输入引用同一个a，梯度对a是二，对x是四x。重复使用同一张量与两个独立但数值相同的张量不同，依赖图必须保留共享节点关系。这个例子能发现最小自动微分引擎中使用赋值代替累加的错误。

### 3. 反向顺序为何是逆拓扑序

一个节点的梯度需要先收集所有使用它的后续节点贡献。拓扑序保证前向依赖先出现，反向沿逆序传播，才能在处理某节点时拿到完整上游梯度。若随意遍历节点，可能在后续贡献尚未到达时就提前传播，遗漏路径。

计算图通常是有向无环图，边表示数据依赖。循环训练程序每步可以构建新的动态图，但单次张量依赖并不因此形成任意循环。若自写引擎遇到图环，需要明确拒绝或采用其他语义，不能让递归无限进行。

## 第三课：从零写一个标量自动微分引擎

~~~python
class Scalar:
    def __init__(self, value, parents=(), backward=None):
        self.value = float(value)
        self.grad = 0.0
        self.parents = tuple(parents)
        self._backward = backward if backward is not None else lambda: None

    @staticmethod
    def wrap(value):
        return value if isinstance(value, Scalar) else Scalar(value)

    def __add__(self, other):
        other = Scalar.wrap(other)
        out = Scalar(self.value + other.value, (self, other))
        def propagate():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = propagate
        return out

    __radd__ = __add__

    def __mul__(self, other):
        other = Scalar.wrap(other)
        out = Scalar(self.value * other.value, (self, other))
        def propagate():
            self.grad += other.value * out.grad
            other.grad += self.value * out.grad
        out._backward = propagate
        return out

    __rmul__ = __mul__

    def backward(self):
        order = []
        visited = set()
        def visit(node):
            if id(node) in visited:
                return
            visited.add(id(node))
            for parent in node.parents:
                visit(parent)
            order.append(node)
        visit(self)
        for node in order:
            node.grad = 0.0
        self.grad = 1.0
        for node in reversed(order):
            node._backward()

x = Scalar(2)
loss = x * x + 3 * x
loss.backward()
assert loss.value == 10
assert x.grad == 7
shared = x * x
loss2 = shared + shared
loss2.backward()
assert x.grad == 8
~~~

### 1. 每个字段的意义

value保存前向数值，grad保存当前反向累计梯度，parents保存直接输入节点，_backward保存局部梯度传播函数。闭包propagate引用输入节点与输出节点，因此反向时能读取输出已累计的grad，再根据局部导数更新输入。

wrap让普通数字也能参与运算，例如三乘x。这个教学引擎没有requires_grad区分，常量节点也会得到梯度，但不会影响示例结果。成熟框架会更细致管理哪些输入需要梯度、哪些状态可释放，以及不同设备与类型的执行。

### 2. 为什么必须用加等于

当self与other恰好是同一节点，例如x乘x，两条局部贡献都应累加到x。第一项给x乘上游，第二项再给x乘上游，总计二x乘上游。若用赋值，第二次写会覆盖第一次，得到一半梯度。

visited只避免拓扑遍历重复访问同一节点，不表示重复边可以忽略。shared加shared的parents中有两个相同引用，局部反向仍执行两次累加，正确得到二倍贡献。节点去重与边贡献计数是不同层次，不能混为一谈。

### 3. 为什么本引擎每次先清零

教学backward把图内全部grad清零，便于单次结果理解。PyTorch叶子梯度通常采用累积语义，用户需要在训练步之间适当清零或设为None。两种设计都可以合理，但接口必须明确，不能把这个小引擎的行为误记成框架默认。

这里没有实现高阶梯度、设备张量、内存释放、版本追踪、线程安全或编译。它只帮助你看见反向图与贡献累加的核心。玩具实现的价值在于可完全理解，不能把它包装成生产自动微分系统。

## 第四课：广播的反向为什么要求和

### 1. 前向复制使用，反向合并贡献

设X为两行三列，bias为长度三，每行都加同一个bias。bias的某个元素被两行使用，因此其梯度是这两行对应位置的上游梯度之和。前向广播看起来没有显式复制，反向仍必须把共享使用产生的贡献合并。

如果bias是单个标量，所有六个输出都依赖它，梯度应求全部上游之和。若bias形状为两乘一，每行一个偏置，则沿列求和，保留行轴。归约轴由原始参数shape与广播方式决定，不是固定“总沿batch求和”。

### 2. 完整unbroadcast参考

~~~python
import numpy as np

def unbroadcast(gradient, original_shape):
    gradient = np.asarray(gradient, dtype=np.float64)
    original_shape = tuple(original_shape)
    while gradient.ndim > len(original_shape):
        gradient = gradient.sum(axis=0)
    for axis, size in enumerate(original_shape):
        if size == 1 and gradient.shape[axis] != 1:
            gradient = gradient.sum(axis=axis, keepdims=True)
    if gradient.shape != original_shape:
        raise ValueError('gradient cannot be reduced to original shape')
    return gradient

g = np.array([[1., 2., 3.], [4., 5., 6.]])
assert np.array_equal(unbroadcast(g, (3,)), [5, 7, 9])
assert np.array_equal(unbroadcast(g, (2, 1)), [[6], [15]])
assert unbroadcast(g, ()).item() == 21
~~~

先合并前向新增的前导轴，再合并原shape中长度一却被扩展的轴。keepdims保留原参数需要的长度一维度，避免最终shape错误。这个函数假定前向广播关系合法，若要作为通用工具，还应在调用前验证原shape与输出shape可广播。

### 3. 共享参数梯度的GPU含义

前向每行独立读取weight或bias，反向却可能让许多program向同一参数梯度写贡献。可以使用partial缓冲再归约、原子加或按参数维重新分配任务。选择影响性能与确定性，但数学目标始终是合并所有共享使用的贡献。

把前向“一行一个program”直接套到weight梯度并普通store，会让最后写入覆盖其他行贡献。这个错误数值上可能看起来像某一行的梯度，使用两行不同上游值就能明确发现。框架不会自动知道自写Kernel遗漏了哪一部分归约。

### 4. 从标量扩展到张量：局部反向返回贡献

标量引擎把梯度直接写进输入节点，张量引擎可以采用另一种清楚结构：每个节点保存一个局部pullback函数，接收该节点上游梯度，返回对每个直接输入的贡献。全局反向引擎负责按逆拓扑序调用它，并把贡献累加到父节点。这样局部数学与全局调度分离。

输入shape与输出shape可能不同，因此每条贡献都要检查shape恰好等于对应父节点shape。广播算子的pullback使用unbroadcast，矩阵乘法使用转置乘法，sum把标量上游广播回输入shape。局部规则各不相同，但全局遍历可以保持统一。

下面的实现只支持float64、加法、逐元素乘法、二维矩阵乘法与全元素sum，不追求完整框架。它足够构建一个小型线性模型损失，展示真实张量反向中的shape、共享节点和梯度累加。代码使用前文unbroadcast，所以应按本章顺序运行，或将该辅助函数一并放入文件。

~~~python
class TensorNode:
    def __init__(self, data, parents=(), pullback=None):
        self.data = np.asarray(data, dtype=np.float64).copy()
        self.parents = tuple(parents)
        self.pullback = pullback
        self.grad = np.zeros_like(self.data)
        self.version = 0
        self.saved_versions = tuple(parent.version for parent in self.parents)

    @staticmethod
    def wrap(value):
        return value if isinstance(value, TensorNode) else TensorNode(value)

    def assign(self, data):
        data = np.asarray(data, dtype=np.float64)
        if data.shape != self.data.shape:
            raise ValueError('assignment shape mismatch')
        self.data[...] = data
        self.version += 1

    def __add__(self, other):
        other = TensorNode.wrap(other)
        def pullback(grad):
            return (unbroadcast(grad, self.data.shape),
                    unbroadcast(grad, other.data.shape))
        return TensorNode(self.data + other.data, (self, other), pullback)

    __radd__ = __add__

    def __mul__(self, other):
        other = TensorNode.wrap(other)
        def pullback(grad):
            return (unbroadcast(grad * other.data, self.data.shape),
                    unbroadcast(grad * self.data, other.data.shape))
        return TensorNode(self.data * other.data, (self, other), pullback)

    __rmul__ = __mul__

    def __matmul__(self, other):
        other = TensorNode.wrap(other)
        if self.data.ndim != 2 or other.data.ndim != 2:
            raise ValueError('two-dimensional matmul only')
        def pullback(grad):
            return grad @ other.data.T, self.data.T @ grad
        return TensorNode(self.data @ other.data, (self, other), pullback)

    def sum(self):
        def pullback(grad):
            return (np.ones_like(self.data) * grad,)
        return TensorNode(self.data.sum(), (self,), pullback)

    def backward(self, seed=None):
        if seed is None:
            if self.data.shape != ():
                raise ValueError('non-scalar output needs explicit seed')
            seed = np.array(1.0)
        seed = np.asarray(seed, dtype=np.float64)
        if seed.shape != self.data.shape:
            raise ValueError('seed shape mismatch')
        order, visited = [], set()
        def visit(node):
            if id(node) in visited:
                return
            visited.add(id(node))
            for parent in node.parents:
                visit(parent)
            order.append(node)
        visit(self)
        for node in order:
            node.grad = np.zeros_like(node.data)
        self.grad = seed.copy()
        for node in reversed(order):
            if node.pullback is None:
                continue
            current_versions = tuple(parent.version for parent in node.parents)
            if current_versions != node.saved_versions:
                raise RuntimeError('saved input changed before backward')
            contributions = node.pullback(node.grad)
            if len(contributions) != len(node.parents):
                raise RuntimeError('wrong pullback arity')
            for parent, contribution in zip(node.parents, contributions):
                contribution = np.asarray(contribution, dtype=np.float64)
                if contribution.shape != parent.data.shape:
                    raise RuntimeError('wrong gradient shape')
                parent.grad += contribution

x_node = TensorNode([[1, 2], [3, 4]])
w_node = TensorNode([[2], [1]])
bias_node = TensorNode([0.5])
prediction = x_node @ w_node + bias_node
loss_node = (prediction * prediction).sum()
loss_node.backward()
assert np.array_equal(prediction.data, [[4.5], [10.5]])
assert np.array_equal(w_node.grad, [[72], [102]])
assert np.array_equal(bias_node.grad, [30])
assert np.array_equal(x_node.grad, [[18, 9], [42, 21]])
~~~

### 5. 完整线性模型反向手算

预测第一行为一乘二加二乘一加零点五，得到四点五；第二行为三乘二加四乘一加零点五，得到十点五。平方损失对预测的上游梯度为九与二十一。weight第一项梯度是一乘九加三乘二十一，得到七十二；第二项是二乘九加四乘二十一，得到一百零二。

bias是长度一向量，广播到两行一列，所以梯度为九加二十一等于三十。输入第一行梯度是上游九分别乘weight二、一，得到十八、九；第二行是二十一乘二、一，得到四十二、二十一。每个结果都能从局部规则与共享使用推导，而不是由框架神秘地产生。

这个例子同时检验矩阵乘法、广播加法、共享节点平方和sum。prediction乘prediction让同一节点作为乘法两侧输入，必须累加两份贡献。若引擎只按节点去重而丢掉重复边，预测梯度会少一半，后续所有参数梯度也少一半。

### 6. 版本计数在防什么错误

乘法反向需要前向时的输入值。若前向后把weight从二、一改成其他值，再直接用新weight计算输入梯度，就不再对应原来的损失函数。version记录通过assign发生的修改，节点创建时保存父节点版本，反向前比较，发现变化就拒绝。

这个教学版本的保护有限：直接修改node.data数组会绕过assign，不会增加version；共享storage视图也未被完整跟踪。我们明确保留这个边界，避免把简化计数器当成成熟框架的完整机制。练习时应只通过公开assign修改，生产框架则需要更系统的存储与版本管理。

保存输入副本可以避免某些修改问题，但增加内存与复制成本。保存引用加版本检查更省复制，却限制反向前的原地修改。重算又要求原始输入仍可用。三种策略各有代价，不能简单说“总是复制”或“永远不允许原地”就完成框架设计。

### 7. 显式种子向量连接向量雅可比积

当输出不是标量时，backward需要指定seed，表示损失对输出的上游梯度。若输出向量有两个元素，seed为一、二，实际求的是这两个输出按一、二加权后的输入梯度。默认把所有输出当作一并不是唯一合理语义，因此教学引擎要求显式提供。

这也解释了框架中对非标量输出直接backward时可能需要gradient参数。自动微分并不是“给我所有导数”这么模糊的请求；它必须知道要把输出空间中的哪一个方向传播回输入。完整雅可比可以通过多次种子选择构造，但大模型通常只需要损失对应的向量雅可比积。

### 8. 张量引擎的边界练习

~~~python
vector_x = TensorNode([2.0, 3.0])
vector_y = vector_x * vector_x
vector_y.backward(np.array([4.0, 5.0]))
assert np.array_equal(vector_x.grad, [16.0, 30.0])

changed_x = TensorNode([2.0])
changed_y = (changed_x * changed_x).sum()
changed_x.assign([3.0])
try:
    changed_y.backward()
except RuntimeError:
    pass
else:
    raise AssertionError('version mismatch was not detected')
~~~

第一个例子验证非均匀上游，第二个验证修改后拒绝。它们分别检查数学规则与状态生命周期，不能互相替代。版本检查通过也不代表梯度公式正确，梯度公式通过也不代表原地修改安全。


## 第五课：PyTorch自定义Function的最小完整例子

~~~python
import torch

class SquareFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * x

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return 2 * x * grad_output

x_torch = torch.tensor([2.0, -3.0], dtype=torch.double, requires_grad=True)
upstream = torch.tensor([4.0, 5.0], dtype=torch.double)
y_torch = SquareFunction.apply(x_torch)
loss_torch = (y_torch * upstream).sum()
loss_torch.backward()
assert torch.equal(x_torch.grad, torch.tensor([16.0, -30.0], dtype=torch.double))
~~~

### 1. 类、静态方法与apply

Function子类定义forward与backward规则，实际调用使用apply，让框架建立对应自动微分节点。staticmethod表示方法不接普通实例self，ctx是框架提供的上下文，用来保存反向需要的信息。不要像普通模块那样实例化后随意直接调用forward，否则可能绕过预期图构建机制。

forward返回真实输出，backward接收对这些输出的梯度，并按forward输入顺序返回对应梯度。若forward有多个张量输入与非张量参数，返回项数量与顺序必须匹配；不求导的参数返回None。错误顺序可能导致shape报错，也可能在相同shape时静默把梯度给错对象。

### 2. save_for_backward与普通属性

需要保存到反向的张量应通过框架提供的保存机制管理，以配合版本检查、生命周期和其他功能。普通标量配置可放在ctx属性中。直接把大量张量塞进普通属性可能绕开某些预期机制，且使内存行为难以审计，具体规则应查当前框架文档。

保存通常不是复制全部数据，而是保留必要引用与状态。若输入在反向前被原地修改，框架可能检测到版本不匹配并报错。不要为了绕过报错使用不安全的数据访问方式；应重新设计保存、复制或重算策略，保持前反向对应同一次计算。

### 3. 上游梯度不一定连续

后续图可能给backward传入非连续梯度视图。若反向直接调用只支持连续输入的Triton Kernel，需要显式检查或转换，并把复制成本纳入训练性能。不能因为前向输入连续，就假设所有反向梯度也连续。

同样，梯度可能是None或在某些接口设置下采用特定物化策略，实际支持方式需遵循框架合同。教学先处理普通单输出张量梯度，扩展多输出或可选梯度时要增加独立测试，而不是在复杂模型中首次发现边界。

### 8. Leaf、非Leaf与梯度存放位置

在动态图中，用户直接创建且需要梯度的参数通常作为叶子节点；运算产生的中间张量带有产生它的梯度关系。框架未必默认把每个中间张量的梯度都长期保存在其grad属性中，因为那会占用额外内存。没有看到某个中间张量grad，不等于梯度没有经过它。

调试时可以用框架支持的保留或钩子机制观察中间梯度，但要意识到这可能改变内存占用与生命周期。正式训练不应为了方便打印而永久保留所有中间状态。先在小图中定位公式，再恢复最小保存集合，能够兼顾理解与资源效率。

叶子梯度累积的语义也需要与本章玩具引擎区分。玩具引擎每次backward重置图内梯度，真实训练框架常让参数grad跨多次backward累加。阅读一段训练代码时，先找到清零、反向和step的位置，再判断某个梯度数值为何比预期大。

### 9. Detach、Clone与转换到NumPy的区别

detach切断自动微分关系，但通常不意味着复制底层数据；clone创建数据副本，但在普通梯度上下文中仍可能保留可微关系。二者组合顺序与用途应明确。想保存一个不参与训练且独立的数据快照，通常需要同时考虑脱离图和复制，而不是只调用一个看起来合适的方法。

把张量转到NumPy或外部库计算，框架未必能追踪其中数学操作。若再把结果包装成新的Tensor，图不会自动重新连接到原输入。自定义Function或custom_op的价值之一，就是为这样的外部计算显式提供反向与注册合同。

不要使用绕过框架保护的数据访问方式来逃避原地修改报错。报错通常提示保存状态与当前数据版本不一致，应该修复生命周期设计。强行绕过可能让程序继续运行，却计算出与前向不对应的梯度，这比明确失败更危险。

### 10. 同一个图能否反向两次

反向结束后，框架可能释放不再需要的保存张量，以降低内存占用。如果你要对同一前向图再次反向，需要按框架接口保留或重新构建图。保留图会增加内存，不能把它作为解决所有报错的默认开关。

两次反向还会涉及梯度累积。即使图被正确保留，第二次可能把新贡献加到已有参数grad上。若目的是比较两种上游梯度，应使用独立图或明确清理梯度，避免把累积结果误当成第二次单独结果。调试代码同样需要遵守训练语义。

重算前向常常比长期保留巨大图更合理，但取决于成本与目的。应明确你需要的是多次向量雅可比积、高阶梯度，还是仅仅重复测试一次函数。不同目标对应不同接口，不能把所有情况都归结为“加一个retain参数”。


## 第六课：RMSNorm的完整一阶反向接入

~~~python
class RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        if x.ndim < 1 or x.shape[-1] == 0:
            raise ValueError('nonempty feature axis required')
        if weight.shape != (x.shape[-1],):
            raise ValueError('weight shape mismatch')
        if not np.isfinite(eps) or eps <= 0:
            raise ValueError('finite positive epsilon required')
        inverse = torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + eps)
        ctx.save_for_backward(x, weight, inverse)
        return x * inverse * weight

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, inverse = ctx.saved_tensors
        u = grad_output * weight
        inner = (u * x).mean(dim=-1, keepdim=True)
        dx = inverse * u - x * inverse.pow(3) * inner
        axes = tuple(range(x.ndim - 1))
        weighted = grad_output * x * inverse
        dweight = weighted.sum(dim=axes) if axes else weighted
        return dx, dweight, None
~~~

### 1. 逐项对应K05公式

inverse是每行逆均方根，shape保留末尾长度一轴。u是上游梯度乘weight，代表先穿过最后的逐元素缩放。inner是每行mean(u乘x)，用于共同尺度路径。dx第一项是直接路径，第二项是归一化尺度随输入变化的修正。

dweight来自输出对weight的局部导数x乘inverse，再乘上游，沿所有前导轴求和。输入若为batch乘time乘hidden，axes为前两个轴；输入若只有hidden一维，axes为空，不能调用默认全轴sum，应保留逐特征梯度。

### 2. 为什么epsilon返回None

eps是普通数值超参数，此接口不对它求梯度，因此backward第三项为None。若将eps改成可训练张量，就改变数学与接口合同，需要推导其梯度并保证正值约束。不能仅把float换成tensor就认为现有backward自动支持。

同理，shape、axis、causal等离散配置通常不是普通可微参数。返回None表示本接口不提供该输入的梯度，不表示其数值对输出毫无影响。可微性与参数影响是不同概念。

### 3. 一阶支持与高阶支持必须分开

上述实现保存forward计算出的inverse，并用于backward。自定义Function的forward内部通常不按普通方式记录用于高阶求导的内部图，因此保存中间inverse可能使二阶梯度路径不完整。它可以是一阶正确实现，却不能未经测试宣称支持gradgrad。

若需要高阶梯度，可以在backward中基于保存的原始x重新计算inverse，使这部分计算在高阶图中可追踪，或采用其他符合框架规则的设计。高阶支持必须用gradgradcheck等独立验证，不能由一阶gradcheck通过推出。

### 4. 只计算需要的输入梯度

某些输入不需要梯度，例如冻结的weight或纯推理输入。自定义反向可以根据框架提供的信息跳过不必要计算，减少内存与运算。但这种优化应建立在正确完整反向之后，且返回位置仍与输入对应，不需要的项返回None。

例如RMSNorm只训练weight而x不需要梯度时，可以省去dx中的共同尺度归约；若只需要x梯度而weight冻结，则可以省去跨行dweight归约。两种路径的资源与Kernel数量不同，性能测试应记录哪些输入需要梯度，不能只说“反向耗时”。

测试应覆盖只求x、只求weight与二者都求三种模式，确认省略路径没有影响保留路径。若复用临时缓冲，要保证跳过某段计算后其他路径不会读取未初始化状态。条件优化不仅影响速度，也改变状态覆盖，需要重新审查。

### 5. 非张量参数与可选输入的返回位置

当forward接收x、weight、bias或None、eps等混合参数，backward返回的槽位仍按原顺序排列。bias为None时对应梯度也应为None，不能简单删除这一项导致后面位置错移。多个输入shape相同时，顺序错误可能不立刻报错，因此需要不同数值与独立梯度参考。

如果输出是多个张量，backward会接收每个输出的上游梯度。某个输出未参与损失时，其梯度处理方式依接口设置而定，应按官方规则支持或限制。一个只为单输出设计的反向，不能直接复制到多输出融合算子而忽略额外路径。

复杂接口可以先写一张参数表，列出类型、是否可选、是否可微、保存需求和对应返回梯度位置。表格不是形式负担，而是避免代码位置错误的直接工具。随后用每次只激活一个输出损失的测试，检查各条反向路径是否正确。

### 6. 内存生命周期与异步设备执行

框架Tensor对象离开Python作用域，不一定意味着底层设备工作已经结束。分配器与stream语义通常负责协调存储复用，但自定义扩展若使用外部缓冲、原始指针或非默认stream，就需要正确记录依赖。不能在Kernel尚未读取完成时释放或复用保存数据。

反向重算尤其依赖前向输入仍然有效。若只保存裸指针而没有持有张量或资源所有权，Python对象可能被释放，地址后来指向别的数据。保存一个数字形式的地址不等于保存数据生命周期。K02的RAII与K13的event/stream在框架扩展中会再次连接起来。

当前教学Function使用框架张量保存机制，不自行管理裸设备内存，降低了这类风险。扩展到C++或外部库时，应明确谁拥有缓冲、何时可释放、哪个stream最后使用它，以及错误路径如何清理。数值测试只覆盖某次时间安排，资源合同还需要结构性证明。


## 第七课：梯度检查从手算到数值差分

### 1. 中央差分检查什么

对标量损失L，输入某元素增加小步长h与减少h，损失差除以二h近似该元素梯度。步长太大有截断误差，太小有消减误差，通常使用double小输入并检查合理步长范围。它不是在任何精度与非光滑点都无条件可靠。

~~~python
def central_difference(function, value, step=1e-5):
    value = np.asarray(value, dtype=np.float64).copy()
    gradient = np.empty_like(value)
    for index in np.ndindex(value.shape):
        original = value[index]
        value[index] = original + step
        plus = float(function(value))
        value[index] = original - step
        minus = float(function(value))
        value[index] = original
        gradient[index] = (plus - minus) / (2 * step)
    return gradient

x_np = np.array([2.0, -3.0])
g_np = np.array([4.0, 5.0])
numeric = central_difference(lambda z: np.sum(z * z * g_np), x_np)
assert np.allclose(numeric, 2 * x_np * g_np)
~~~

### 2. PyTorch gradcheck的使用

~~~python
x_check = torch.randn(2, 5, dtype=torch.double, requires_grad=True)
w_check = torch.randn(5, dtype=torch.double, requires_grad=True)
assert torch.autograd.gradcheck(
    lambda x, w: RMSNormFunction.apply(x, w, 1e-5),
    (x_check, w_check))
~~~

这里同时检查x与weight，不能只检查一个输入。使用不同随机上游或内部检查方式可覆盖向量雅可比积。若测试失败，先在两三个元素上手算，再分别检查直接路径、共同尺度路径与参数归约，不要立即放宽容差。

gradcheck只检查数值梯度关系，不证明别名、mutation、fake或编译注册正确。它也不自动证明所有shape、dtype与设备路径正确。一个CPU double版本通过，GPU float32仍需要自己的前后向误差与边界验收。

### 3. 故意删除一项验证检查器

把RMSNorm的dx暂时改为inverse乘u，遗漏共同尺度修正。选择非零且非对称输入，gradcheck应失败。若仍通过，检查是否实际调用了修改后的实现、输入是否退化、测试是否吞掉异常。

错误注入应在独立练习分支或CPU参考中进行，恢复后保留能抓住该错误的测试。它的目的不是制造风险，而是确认测试有区分力。一个从未见过失败的检查器，未必真的检查了你以为的条件。

### 4. 用方向导数减少大型检查的成本

逐元素中央差分需要对每个输入位置扰动两次，输入很大时成本高。可以选择一个固定随机方向v，比较损失沿x加hv与x减hv的差分，与解析梯度和v的内积。这个方向检查一次覆盖许多元素，适合在小型逐元素检查通过后增加较大shape验证。

方向检查不能证明每个分量都正确，某些错误可能在所选方向上抵消，因此应使用多个独立方向，并保留小尺寸完整逐元素检查。它是补充证据，不是为了节省成本而取消最基本的梯度验收。

对于矩阵乘法，还可以验证局部伴随关系：前向微小扰动产生的输出变化与上游的内积，应等于输入扰动与反向梯度的内积。这个关系直接来自链式法则，能检查转置与归约轴。使用不同形状和非对称数据，可以让错误更难偶然抵消。

测试时保持随机方向固定、使用double、选择合理步长，并报告两边标量及差值。若误差随步长缩小先改善后恶化，可能是截断与舍入的正常权衡；若始终相差固定比例，优先检查loss缩放、遗漏上游或求和与均值混用。

## 第八课：自定义算子的注册合同

### 1. Schema描述公开边界

注册算子需要名字空间、名称、参数与返回类型，以及mutation等信息。名称空间避免与其他扩展冲突；类型信息帮助框架理解调用结构；副作用声明帮助编译与功能化分析。数学结果正确但副作用声明错误，会让上层基于错误假设重排或复用数据。

一个算子可以在多个设备后端注册实现，但所有实现应遵守同一公开数学与元信息合同。若CPU输出连续而CUDA输出某种特殊stride，接口是否允许这种差异要明确，fake实现也要匹配目标行为。不能只靠相同shape声称完全一致。

### 2. 完整平方注册例子

~~~python
@torch.library.custom_op('llm_course_deep::square', mutates_args=())
def registered_square(x: torch.Tensor) -> torch.Tensor:
    return (x * x).contiguous()

@registered_square.register_fake
def registered_square_fake(x):
    return torch.empty(x.shape, device=x.device, dtype=x.dtype)

def square_setup_context(ctx, inputs, output):
    (x,) = inputs
    ctx.save_for_backward(x)

def square_backward(ctx, grad_output):
    (x,) = ctx.saved_tensors
    return 2 * x * grad_output

registered_square.register_autograd(
    square_backward, setup_context=square_setup_context)
~~~

真实实现显式返回连续结果，fake也创建相同shape、device与dtype的连续输出，减少非连续输入下元信息不一致的歧义。名字空间与原始教学lab不同，避免在同一进程重复注册同名算子。重复运行Notebook单元时仍要注意注册对象生命周期与重名问题。

mutates_args为空表示不修改任何输入。真实实现只读x并创建结果，符合这一声明。若改成x.mul_(x)原地平方，却不更新mutation合同，就会破坏框架推理。不能为了省一次分配而偷偷改变副作用。

### 3. Fake实现不能读取真实数据

fake用于只凭元信息推断输出，不应调用item读取输入值，不应根据真实内容决定shape，也不应实际访问外部设备内存。shape、stride、dtype、device和布局关系是它的主要输入。若输出shape依赖数据，需要使用框架支持的动态输出机制或重新设计接口，不能伪造一个固定shape。

真实实现与fake应在各种受支持输入上匹配元信息。非连续输入特别重要，因为empty_like可能保留某种布局，而真实运算结果未必相同。这里通过明确连续输出简化合同，代价是可能发生必要重排，性能记录要包含它。

## 第九课：Dispatch与不同执行模式

### 1. 调度选择不是普通if的全部

框架调度根据张量设备、类型、布局以及自动微分等执行上下文选择相应实现或包装层。它让同一个公开算子名可以服务不同后端，同时保持统一接口。具体调度键与优先级是框架实现细节，阅读源码时需要按固定版本追踪，不能凭一个简单if模型推断全部顺序。

教学上可以先想成一个注册表：算子名与后端组合映射到实现函数。再逐步加入autograd、fake与其他变换层，理解它们可能在真正设备Kernel前后参与处理。这个简化模型帮助建立层次，但不代表真实dispatcher只有一个字典查找。

### 2. Eager、编译与导出各检查不同问题

Eager直接执行当前调用路径，容易观察数值与异常。编译模式需要捕获或分析算子图，可能依赖fake与副作用合同。导出通常还要求更明确的图与形状约束。某算子在eager能跑，不自动意味着在编译或导出中可用。

对自定义算子，应分别测试真实前向、梯度、fake元信息与目标编译模式。若某模式尚未验证，文档写明未验证；若明确不支持，写明不支持。两者含义不同，不能用一个模糊标签覆盖。

### 3. Opcheck与Gradcheck互补

opcheck用于检查注册合同与相关框架子系统的配合，gradcheck用于数值梯度。二者都不替代独立前向oracle，也不证明全部性能或分布式行为。一个完善验收会同时包含前向值、梯度、注册与所需执行模式，而不是寻找一个万能检查函数。

~~~python
x_registered = torch.randn(2, 3, dtype=torch.double, requires_grad=True)
torch.library.opcheck(registered_square, (x_registered,))
assert torch.autograd.gradcheck(registered_square, (x_registered,))
transposed = torch.randn(3, 2, dtype=torch.double).T.requires_grad_()
torch.library.opcheck(registered_square, (transposed,))
assert registered_square(transposed).is_contiguous()
~~~

这些调用需要实际安装支持相应API的PyTorch版本，当前环境未执行。若版本不支持，应明确报告API缺失并按官方文档适配，不能把未运行当成通过。源码示例与运行证据始终分开。

## 第十课：三个故障案例与定位

第一类故障是平方反向忘乘上游梯度。损失直接sum(y)时可能通过，因为上游全一；改成与不同权重g的内积，错误立刻暴露。修复后保留非均匀上游测试，避免未来再次退化。

第二类故障是RMSNorm权重梯度只沿batch求和，遗漏time。使用输入二乘三乘五，weight五，正确dweight必须长度五。若得到三乘五，shape已错；若后续又被隐式广播或累加，可能进一步掩盖。先检查shape，再检查数值。

第三类故障是真实算子返回连续输出，fake却按非连续输入保留stride。eager前向可能完全正确，编译或opcheck才失败。修复应统一输出布局合同，并测试转置输入，而不是把fake改成随意shape相同就结束。

## 第十一课：十二道练习与完整答案

### 练习一：上游梯度手算

题目：x为三，y为x平方，损失五y。答案：局部导数六，上游五，输入梯度三十。只返回六遗漏链式法则的后半段。

### 练习二：共享节点梯度

题目：a=x平方，L=a+a，x为二。答案：对a梯度二，对x梯度八。拓扑节点只访问一次，但两条使用边都贡献梯度。

### 练习三：广播参数梯度

题目：上游矩阵[[一,二],[三,四]]，bias长度二。答案为四、六；若bias是标量则十；若bias形状二乘一则三、七并保留列轴。

### 练习四：实现乘加反向

本题限定逐元素公式`y = x * weight + bias`，不是矩阵乘法。x的形状就是输出形状，grad与x同形状；weight和bias都是长度等于x最后一维的一维向量，只在前导轴上广播。函数不需要bias的数值，因为对bias的局部导数恒为一；但它使用weight.shape还原bias梯度，依赖二者同形状这一前提。如果允许x本身被广播，dx也必须还原到x原形状；如果bias采用独立形状，函数应显式接收该形状并据此计算db，不能直接套用本题受限接口。

~~~python
def affine_backward(x, weight, grad):
    x = np.asarray(x, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    grad = np.asarray(grad, dtype=np.float64)
    dx = grad * weight
    dw = unbroadcast(grad * x, weight.shape)
    db = unbroadcast(grad, weight.shape)
    return dx, dw, db

dx_affine, dw_affine, db_affine = affine_backward(
    [[1, 2], [3, 4]], [2, 3], [[1, 1], [2, 2]])
assert np.array_equal(dx_affine, [[2, 3], [4, 6]])
assert np.array_equal(dw_affine, [7, 10])
assert np.array_equal(db_affine, [3, 3])
~~~

### 练习五：返回顺序

题目：forward接收x、weight、eps，backward应返回什么顺序。答案为dx、dweight、None，数量与输入对应，不能把参数梯度与输入梯度交换。

### 练习六：高阶梯度限制

答案：一阶正确不代表高阶正确，保存forward内部中间量可能缺少高阶依赖。需要重算可追踪表达式或采用正确设计，并独立运行二阶检查。

### 练习七：新输出与别名

答案：若接口承诺新输出，直接返回输入或其共享storage视图违反合同。数值相同不能替代存储语义，后续原地写入与编译优化会受影响。

### 练习八：Fake数据依赖

答案：不能用x.item决定返回长度，因为fake没有真实数据。若输出长度依赖内容，需要使用支持动态输出的机制或改变接口，不能猜一个长度。

### 练习九：Opcheck能否替代Gradcheck

答案：不能。注册与子系统检查和数值导数检查分别覆盖不同问题，还需要独立前向参考与边界测试。

### 练习十：梯度累积

答案：多个损失或微批次可向同一叶子参数累积梯度，训练步之间是否清零由训练逻辑决定。不能把意外累积当成算子反向错误，也不能每次反向都无条件清零破坏合法微批次累积。

### 练习十一：非连续上游

答案：backward收到的梯度未必连续。受限Kernel应显式转换或支持stride，并把复制纳入性能与内存记录，不能只检查前向输入布局。

### 练习十二：完整集成验收

答案包含前向值、各输入梯度、非连续与边界输入、别名与mutation、fake元信息、opcheck、所需编译模式及实际设备路径。未执行项明确列出，不从某一项通过推断其他项。

## 第十二课：亲手搭一个元信息与调度注册表

### 1. 元信息是计算前就能知道的事实

shape、dtype、device与stride描述张量如何被解释和存放，不包含每个元素的具体数值。许多输出元信息只依赖这些属性，例如平方保持shape，矩阵乘法输出行数来自左输入、列数来自右输入。编译器可以在不执行真实数据计算时利用这些关系规划图与内存。

但并非所有算子输出shape都只依赖输入shape。筛选非零元素、按数据条件压缩序列等操作，输出数量可能依赖内容。此时需要符号动态尺寸或明确的上界与额外长度输出等设计。不能假装所有算子都有一个简单固定shape公式，否则fake实现会与真实运行不一致。

stride同样属于元信息。输出连续与输出转置视图可能shape一样，却在后续访问和别名上不同。框架编译不能只看行列数量，还要知道布局与副作用，才能安全安排重排、融合或缓冲复用。K01的地址知识在这里变成编译合同。

### 2. 用数据类表达教学元信息

~~~python
from dataclasses import dataclass

@dataclass(frozen=True)
class TensorSpec:
    shape: tuple
    dtype: str
    device: str
    strides: tuple

def contiguous_strides(shape):
    result = [0] * len(shape)
    stride = 1
    for axis in range(len(shape) - 1, -1, -1):
        result[axis] = stride
        stride *= shape[axis]
    return tuple(result)

def spec_of(array):
    array = np.asarray(array)
    return TensorSpec(array.shape, str(array.dtype), 'cpu',
                      tuple(value // array.itemsize for value in array.strides))

def square_spec(input_spec):
    return TensorSpec(input_spec.shape, input_spec.dtype,
                      input_spec.device,
                      contiguous_strides(input_spec.shape))
~~~

这个教学Spec把device简化成字符串，stride以元素为单位。真实框架还有符号尺寸、布局类型、稀疏信息、内存格式与别名关系等更多内容。我们先保留足够解释本章平方与矩阵乘法的字段，避免把简化模型误称为完整Tensor元数据。

contiguous_strides从最后轴开始累乘，体现行优先连续布局。零尺寸与长度一维在真实框架中可能有多个等价stride表示，教学检查若要求精确相等，应限制或明确这些情况。元信息一致性并不总是简单比较一个固定公式，接口需要说明允许的布局等价范围。

### 3. 注册表分离名字、元信息与真实实现

~~~python
class OperatorRegistry:
    def __init__(self):
        self.schemas = {}
        self.kernels = {}

    def define(self, name, infer_spec, mutates_inputs=False):
        if name in self.schemas:
            raise ValueError('operator already defined')
        self.schemas[name] = (infer_spec, mutates_inputs)

    def implement(self, name, device, function):
        if name not in self.schemas:
            raise ValueError('define schema before implementation')
        key = (name, device)
        if key in self.kernels:
            raise ValueError('kernel already registered')
        self.kernels[key] = function

    def infer(self, name, *specs):
        return self.schemas[name][0](*specs)

    def call(self, name, *arrays):
        specs = tuple(spec_of(array) for array in arrays)
        devices = {spec.device for spec in specs}
        if len(devices) != 1:
            raise ValueError('mixed devices not supported')
        device = next(iter(devices))
        expected = self.infer(name, *specs)
        snapshots = tuple(np.asarray(array).copy() for array in arrays)
        result = self.kernels[(name, device)](*arrays)
        actual = spec_of(result)
        if actual != expected:
            raise AssertionError(f'metadata mismatch: {actual} != {expected}')
        if not self.schemas[name][1]:
            for before, after in zip(snapshots, arrays):
                if not np.array_equal(before, after):
                    raise AssertionError('undeclared input mutation')
            for array in arrays:
                if np.shares_memory(result, array):
                    raise AssertionError('fresh output contract violated')
        return result

registry = OperatorRegistry()
registry.define('course.square', square_spec)
registry.implement('course.square', 'cpu',
                   lambda x: np.ascontiguousarray(np.asarray(x) * np.asarray(x)))
input_square = np.arange(6, dtype=np.float64).reshape(2, 3).T
output_square = registry.call('course.square', input_square)
assert np.array_equal(output_square, input_square * input_square)
assert output_square.flags.c_contiguous
~~~

这个call会复制输入做副作用检查，成本很高，适合教学验证而不是生产快速路径。真实框架不能每次都复制所有输入来确认合同，只能依赖正确注册、版本机制与专门检查工具。因此注册声明必须可信，不能把它当作装饰性注释。

这里把“不修改输入”与“返回新输出”合并成简化合同，真实系统中两者是独立属性：一个只读算子可以合法返回视图，前提是schema正确声明别名。教学选择新输出是为了让检查更直观，不代表所有不修改输入的算子都必须复制。

### 4. 故意注册错误实现观察哪层失败

若真实平方实现返回输入x，元信息可能完全匹配，但数值与新输出合同错误；若原地平方再返回x，既修改输入又别名；若返回转置结果，shape或stride可能不匹配；若只返回全零连续数组，元信息检查通过，数值oracle才会失败。

这四个反例说明注册检查、别名检查与数学检查互补。一个系统能证明shape与stride正确，不代表算术正确；一个数值测试通过，不代表副作用声明正确。成熟的自定义算子验收需要多层证据，而不是寻找唯一万能工具。

可以把错误实现注册到独立名称，使用最小输入并预期抛出异常，避免污染正确实现。错误注入后恢复注册表或新建实例，保证测试之间没有隐藏状态依赖。全局注册机制尤其要注意测试隔离，否则重复定义与缓存可能干扰结果。

### 5. 矩阵乘法的元信息推导

设左Spec为M乘K，右Spec为K乘N，dtype与device一致，输出Spec为M乘N连续。推导函数只需要检查维数、内维与类型，不需要读取任何数值。即使输入是转置视图，逻辑shape仍决定数学输出，真实Kernel是否支持该stride则是实现路径选择问题。

可以把“数学支持的输入”与“某个Kernel支持的输入”分成两层。schema允许任意合法矩阵布局，dispatcher选择连续专用Kernel或通用strideKernel；若没有可用实现，则明确报不支持。不要为了某条快速路径方便，把所有数学能力混成一个难以扩展的接口。

输出dtype也可能由输入类型提升规则决定，例如混合类型计算。教学先限制相同dtype，避免引入完整类型系统。扩展时应把提升规则集中定义并在真实实现、fake与测试中共用，不能三处各凭直觉写一遍而产生分歧。

### 6. 调度键只是选择条件的一部分

真实框架可能在设备Kernel之外插入autograd、自动混合精度、函数变换或功能化等层。某些层处理完自己的责任后继续向下分派，而不是直接完成全部计算。理解这种层次有助于阅读dispatcher代码中看似重复的包装与再分派。

不要把“同一个算子有CPU和CUDA实现”理解成用户必须手动调用两个不同函数。公开接口保持一致，输入元信息与上下文决定路径。相反，若某个自定义Python函数内部自己随意选择设备并搬运数据，会隐藏成本和副作用，破坏用户对设备执行的预期。

阅读真实源码时，先找到schema定义，再找后端注册，再找自动微分或组合规则，最后进入具体Kernel。沿一条具体输入路径走通，比一开始遍历所有dispatch key更容易形成代码级理解。每一步记录函数名、输入元信息与下一层调用，形成可复查的调用链。

## 第十三课：保存、重算、高阶梯度与训练步骤

### 1. 保存状态的最小充分集合

反向需要什么应由公式推导。平方需要原始x或能恢复x的信息；softmax可用输出概率；RMSNorm需要x、weight与逆尺度或可重算尺度；矩阵乘法需要A与B；在线attention需要Q、K、V、输出及归一化统计量等。保存全部前向中间值通常简单但耗内存，盲目只存输出又可能无法恢复必要信息。

“最小充分”还依赖数值与接口。某些函数从输出无法唯一恢复输入，例如平方丢失符号，因此不能仅保存y再用平方根还原x。某些函数理论可逆，但数值反演不稳定或比重算更贵。保存策略必须同时考虑数学、精度和成本。

### 2. 重算不是免费，也不是错误

RMSNorm的inverse每行一个值，保存成本很小；重算需要再次读取x、平方、归约与开根。是否值得取决于输入规模、内存压力和框架执行方式。在线attention重算分数则可能节省巨大的概率矩阵，收益结构不同。不能把某个算子的选择推广成所有算子都应重算。

重算必须使用与前向一致的参数和输入版本。若训练代码在反向前更新weight，或某个缓冲被复用，重算就会对应另一函数。优化器通常在反向之后更新参数，正是为了保持这条依赖顺序；异步与流水线训练需要更细致地维护版本。

### 3. 完整支持高阶路径的RMSNorm教学版本

~~~python
class RMSNormRecompute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        ctx.save_for_backward(x, weight)
        ctx.eps = float(eps)
        inverse = torch.rsqrt((x * x).mean(-1, keepdim=True) + eps)
        return x * inverse * weight

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        inverse = torch.rsqrt((x * x).mean(-1, keepdim=True) + ctx.eps)
        u = grad_output * weight
        dx = inverse * u - x * inverse.pow(3) * (u * x).mean(-1, keepdim=True)
        axes = tuple(range(x.ndim - 1))
        product = grad_output * x * inverse
        dw = product.sum(dim=axes) if axes else product
        return dx, dw, None

# Run only in an environment with a compatible PyTorch installation.
x_second = torch.randn(2, 3, dtype=torch.double, requires_grad=True)
w_second = torch.randn(3, dtype=torch.double, requires_grad=True)
assert torch.autograd.gradcheck(
    lambda x, w: RMSNormRecompute.apply(x, w, 1e-4),
    (x_second, w_second))
assert torch.autograd.gradgradcheck(
    lambda x, w: RMSNormRecompute.apply(x, w, 1e-4),
    (x_second, w_second))
~~~

这里在backward中重新计算inverse，使高阶求导时这段依赖可以被记录。示例仍需实际执行检查，不能因为结构看起来合理就声称已经支持所有高阶场景。生产版本还要复制前一个wrapper的输入验证，处理所需dtype与布局，并说明是否支持函数变换。

高阶梯度不只是学术要求，有些优化、元学习、隐式微分或梯度惩罚会用到。若你的算子只承诺一阶训练，可以明确限制，而不是勉强提供错误二阶结果。能力边界清楚比模糊宣称“支持autograd”更有价值。

### 4. 梯度累积与优化器更新

一个训练步骤通常先清理或准备梯度，执行前向与损失，反向累积参数梯度，再由优化器更新参数。多个微批次可以在一次更新前累积梯度，这是一种有意行为。若每个微批次都清零，就破坏累积；若每步忘记清理，又可能意外把历史梯度叠加。

梯度是当前损失对参数的导数，优化器状态还可能包含动量、二阶统计等。自定义算子负责正确产生梯度，不应在backward里偷偷更新参数，否则会混淆图依赖与优化步骤，也可能破坏重入或高阶计算。保持职责分离有助于调试。

### 5. 用一阶参数更新而不是长期loss验证集成

构造同一输入、相同初始weight和相同损失，分别使用原生表达式与自定义算子。比较前向输出、损失、输入梯度、参数梯度，最后用同一学习率更新一次参数并比较。若这些都一致，再考虑多步训练。长期loss下降可能掩盖局部梯度错误，尤其模型有许多其他可训练参数时。

~~~python
def compare_one_training_step():
    torch.manual_seed(10)
    x0 = torch.randn(2, 4, dtype=torch.double)
    w0 = torch.randn(4, dtype=torch.double)
    target = torch.randn(2, 4, dtype=torch.double)
    x_ref = x0.clone().requires_grad_()
    w_ref = w0.clone().requires_grad_()
    x_custom = x0.clone().requires_grad_()
    w_custom = w0.clone().requires_grad_()
    inv_ref = torch.rsqrt(x_ref.square().mean(-1, keepdim=True) + 1e-5)
    y_ref = x_ref * inv_ref * w_ref
    y_custom = RMSNormFunction.apply(x_custom, w_custom, 1e-5)
    loss_ref = (y_ref - target).square().mean()
    loss_custom = (y_custom - target).square().mean()
    loss_ref.backward()
    loss_custom.backward()
    torch.testing.assert_close(y_custom, y_ref)
    torch.testing.assert_close(x_custom.grad, x_ref.grad)
    torch.testing.assert_close(w_custom.grad, w_ref.grad)
    updated_ref = w_ref.detach() - 0.01 * w_ref.grad
    updated_custom = w_custom.detach() - 0.01 * w_custom.grad
    torch.testing.assert_close(updated_custom, updated_ref)

# 只在已经准备好兼容 PyTorch 的环境执行；定义函数本身不会运行上述断言。
compare_one_training_step()
~~~

clone建立独立参数，避免两条路径共享同一个grad缓冲。detach用于构造更新后的数值比较，不让检查过程本身进入新的训练图。两个loss都采用mean，因此上游缩放相同；若一边sum、一边mean，梯度会相差元素数，不应误诊为算子公式错误。

### 6. 非光滑点与有限差分的限制

ReLU在零点不可微，有限差分可能得到与框架选择的次梯度不同的数值。最大值在平局处也存在选择问题，top-k与离散索引更有非连续行为。梯度检查应避免把不可微边界当成普通光滑点，或明确验证框架所选合同。

这不意味着可以忽略边界。应把“光滑区域数值梯度检查”和“非光滑点约定检查”分开。前者验证导数公式，后者验证接口在平局、零点或mask切换时的定义。将两种问题混在一起，会导致盲目调容差或错误修改正确实现。

### 7. 自动混合精度需要单独设计

自动混合精度可能改变进入算子的dtype或某些运算的执行精度。自定义算子若只支持float32，应明确拒绝或通过受支持方式处理转换；若支持半精度输入与单精度累加，要说明输出类型与保存状态精度。不能假设框架会自动为任意外部Kernel选择正确策略。

反向同样要考虑梯度类型、缩放与溢出。损失缩放是训练系统处理数值范围的一种机制，不会修复错误的梯度公式，也不会让不受支持的输入类型变合法。先在double或float32建立正确基线，再逐步加入混合精度，是更可解释的路线。

## 第十四课：编译、函数变换与框架源码阅读

### 1. 图捕获依赖可观察的合同

编译器需要知道算子输入输出与副作用，才能决定哪些节点可以融合、重排或消除。若算子声明纯函数却修改输入，编译器可能合法地按错误声明优化，导致结果与eager不同。问题根源是合同不真实，而不是编译器“随机出错”。

输出别名也会影响缓冲复用。若输出共享输入storage，后续对输出写入可能改变输入；若编译器以为它是新分配，就可能错误延长或缩短某些数据生命周期。别名分析不是抽象术语，而是直接关系到内存安全与数值结果。

### 2. 功能化为何关心原地操作

某些编译与变换流程会把原地更新改写成更易分析的函数式形式。要做到这一点，必须知道哪些参数被修改、修改后的值如何返回或被后续使用，以及视图之间如何共享storage。错误mutation声明会让这个转换失去依据。

初学自定义算子时，优先采用新输出、只读输入的接口，可以减少这层复杂度。需要原地操作时，再明确schema与测试别名关系，而不是为了省内存先写原地版本后补文档。性能收益应该建立在正确合同上。

### 3. 动态shape与数据依赖分支

编译器可能用符号表示batch或序列长度，而不是为每个尺寸固定展开。fake实现应尽量使用shape关系推导输出，不应把符号尺寸强制转成真实数据值。某些Python分支或标量提取会导致图中断或额外特化，具体行为依版本与编译模式而定。

例如矩阵乘法输出shape只依赖输入维度，容易描述；根据x正元素个数分配输出则依赖数据，需要不同机制。把两类算子混为一谈，会让fake代码在普通输入上看似可用，却在编译追踪时失败。先分类输出shape来源，再选择接口设计。

### 4. Vmap与批处理规则

自动批处理变换会在算子外增加或解释批次轴。一个只支持二维矩阵的自定义实现，未必能自动处理额外batch维。需要组合规则、显式批处理实现或受支持的变换注册，具体接口以当前版本文档为准。

不要因为手动把batch循环写在Python外层能运行，就宣称算子支持所有函数变换。vmap、前向模式自动微分与高阶反向各有不同要求。文档应列出实际验证的功能，而不是用“框架兼容”四个字覆盖全部。

### 5. 前向模式与反向模式的方向

反向模式传播输出损失对输入的向量雅可比积，适合少量标量输出对应大量参数的训练。前向模式传播输入扰动如何影响输出，即雅可比向量积。自定义反向规则不自动提供前向模式规则，两者需要各自数学与接口支持。

对于平方，反向是g乘二x，前向扰动传播是二x乘输入切向量，形式相似；对于一般多输入多输出函数，组织不同。学习时可以先手算一个小例子，理解“上游梯度”和“输入方向”不是同一种对象，再查具体注册方式。

### 6. 进入框架源码的具体路线

选一个简单算子，例如加法或RMSNorm，先从公开Python调用找到对应算子名，再查schema与后端注册，接着定位CPU或CUDA实现，最后追踪autograd规则与保存状态。每次只选择一个dtype、一个设备和一个shape，避免同时理解全部分支。

记录每层输入输出：Python层看到Tensor对象，dispatcher层看到算子与调度信息，后端实现看到张量元信息与数据指针，Kernel看到具体地址与尺寸，autograd节点保存反向所需状态。这样你能把高层框架与前面学过的低层算子连接起来。

框架源码包含大量生成代码、宏与模板，不能只搜索一个函数名找不到就停止。先阅读该项目固定版本的代码生成与注册说明，再追踪生成声明的来源。学习目标不是背文件路径，而是理解一次调用怎样穿过层次并到达执行实现。

### 7. C++注册的概念与Python注册对应

C++扩展同样需要定义算子schema并为相应后端注册实现。它可能直接调用ATen已有操作，也可能调用自写CUDA Kernel。无论实现语言如何，别名、mutation、autograd与元信息责任仍然存在；把代码从Python移到C++不会自动获得这些能力。

读取C++注册代码时，区分定义名字与签名的部分、选择后端的部分、真正函数体以及编译构建入口。很多初学者看到注册宏就以为那里完成了计算，实际它只是把函数连接到调度系统。沿被注册函数继续追踪，才能理解代码级行为。

### 8. 自定义边界可能限制融合

一个不透明自定义算子可以保护外部实现的边界，但编译器可能无法穿透它与前后操作一起优化。若内部只是几个现有框架操作，过早包装成黑盒可能损失融合机会。选择自定义算子时应权衡封装、外部实现需求与编译可见性。

反过来，某些专用Kernel已经把关键步骤高效融合，作为自定义边界可能很有价值。判断不能只凭“自定义更底层所以更快”，而要比较完整执行图、内存流量、调用开销与数值合同。框架集成与Kernel优化必须一起评估。

### 9. 一份框架兼容性测试表

至少列出eager前向、eager反向、gradcheck、必要时gradgradcheck、fake元信息、opcheck、目标compile模式、非连续输入、空输入与所需dtype。每项标明支持、明确不支持或未执行，并附具体命令与环境。不要用一个总的绿色勾号代替不同能力。

如果某项失败，先判断是数学错误、注册错误、模式未支持还是环境缺失，再决定修复或收窄合同。收窄合同应诚实记录，不应悄悄跳过失败测试。一个受限但完整验证的算子，比范围模糊的“通用实现”更可靠，也更容易逐步扩展。


### 10. 三种不同的“图”不要混成一件事

自动微分图记录数学依赖与反向规则；编译图表示可分析和优化的算子程序；CUDA Graph记录设备任务及依赖以便重放。它们可以在同一系统中协作，但目的、生命周期和限制不同。一个算子支持autograd，不代表自动支持所有编译捕获或CUDA Graph重放。

自动微分关注哪些输入影响损失以及如何求导，编译关注如何保持语义并改进执行，CUDA Graph关注减少提交开销并复用任务结构。看到错误中的graph一词时，先判断是哪一层，才能查对文档和排查方向。否则很容易拿保存梯度图的参数去解决设备图捕获问题。

### 11. 一次源码阅读怎样算真正完成

选择一个算子后，能够写出从公开调用到后端实现、再到反向节点的实际路径，并解释每个保存张量为何必要。接着修改一个受控参数或添加一个小型测试，预测哪一层会变化，再运行验证。仅找到文件并复制函数名，不足以证明理解了调用机制。

阅读笔记应包含固定commit、符号名称、输入合同、关键状态与一个可运行小例子。框架版本升级后，路径可能变化，但这些问题仍然有效。你的目标是获得重新定位与验证的能力，而不是永远依赖某个版本的行号。


## 第十五课：小项目——让RMSNorm进入一个真实训练步骤

先用框架原生表达式建立参考，再用自定义Function替换，使用相同初始参数、输入和上游损失，比较第一步输出、输入梯度、weight梯度与更新后参数。确认这些一致后，再运行少量训练步观察损失，而不是直接训练很久才判断成败。

第二阶段用K09的Triton前向替换Function前向，保留清晰框架反向，重新验证dtype、连续性与保存状态。第三阶段才考虑把反向也写成Kernel，分别设计每行dx与跨行dweight归约。每次只改变一个层次，保留可回退基线。

最终交付包括算子合同、完整前后向、保存与重算说明、数值与注册检查、目标执行模式结果和已知限制。能够解释这些内容，才说明你理解了Kernel如何成为框架中的可靠组件。

请把数学导数、框架注册与设备执行分别验收；三者都正确，算子才真正完成从公式到训练系统的集成。
