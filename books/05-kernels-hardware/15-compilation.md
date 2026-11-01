# K15　从表达式到机器指令：亲手构造、观察和调试编译链

本章先修是K02的对象、类型与编译链接，K03的线程索引，K08的Program与编译期常量。你不需要先读完编译原理教材。我们从一个只有加、减、乘的表达式语言开始，自己写解析、解释执行、形状检查、常量折叠、公共子表达式消除和指令执行器，再把这条具体过程对应到C++、CUDA与Triton。每出现一个新层次，都说明它接收什么、输出什么、保持什么不变、可能在哪一步失败。

本章Python代码按照出现顺序放入同一个文件运行，使用标准库和NumPy；它们是能执行的教学编译器，不生成真正GPU指令。主机C++实验有完整程序，可以在Linux或WSL中用g++运行。CUDA命令与设备产物阅读另行标明，当前交付环境没有nvcc和GPU，不能将前两类实验的成功解释为CUDA编译成功。先把这些证据分开，才能建立可靠的底层学习方法。

## 第一课：编译器究竟把什么变成什么

源代码是供人写程序的语言表示。机器并不直接理解“把矩阵按行归一化”这个意图，它需要明确的操作、数据地址、控制流程和调用约定。编译器把一种程序表示变成另一种，并在规定的语言语义下保持可观察行为。这里的“可观察”包括返回值、对外部存储的写入、允许的异常与副作用，不仅是某个中间变量看起来还存在。

解释器则读取程序表示并直接执行对应行为。例如后面我们的解释器遇到乘法节点，就调用NumPy乘法。编译器可以先把整棵树变成一列较低层操作，之后由机器或另一解释器执行。真实系统常混合两者：Python解释执行主机控制逻辑，遇到Triton函数时触发编译，再启动设备程序。因而“用了Python”与“没有编译”不存在必然关系。

先看表达式“先算x乘a加b，再把结果平方”。输入x为数组[1,2,3]，a为标量2，b为标量1，结果是[9,25,49]。这个程序至少存在三种表示：人写的字符串“(x*a+b)*(x*a+b)”；以乘法为根的语法树；先算t0=x*a、t1=t0+b、t2=t1*t1的指令表。三者语法不同，输入输出关系相同，但重复工作的数量可能不同。

编译前端负责理解源语言，包括词法、语法、名称和类型。词法分析把字符分成名字、数字、运算符、括号等记号；语法分析把记号组织成树；名称解析判断x指向哪个变量；类型检查确认这些操作对输入是否合法。中间表示通常简称IR，目的是让后续处理不用反复面对复杂源语言。后端把IR映射到目标指令、寄存器和机器约束。

这些职责是概念分层，不是每个项目必须拥有四个同名目录。实际编译器可以有多级IR，某些检查在不同阶段重复进行。框架先处理整个算子图，Triton再处理单个块程序，LLVM后端继续降低表示，设备汇编器再完成目标编码。看到一长串编译调用栈，首先标出当前表示与错误所属层，不要把所有错误统一归结为“CUDA坏了”。

还有一个容易混淆的词是调度。编译器的指令调度安排依赖允许的指令次序；运行时调度选择调用哪个已编译内核；GPU硬件调度选择就绪Warp；推理服务调度决定哪些请求进入本轮批次。它们通过数据和产物连接，但解决的问题不同。请求队列延迟很大时，只检查汇编中的乘法指令不会直接回答问题。

我们给教学语言设定清晰边界：只有有限的float32常量、输入变量、加法、减法、乘法、一元负号与括号；输入可以是float32标量或相同形状数组；不支持循环、函数调用、文件操作、原地写入和隐式任意广播。缩小语言不是逃避困难，而是让每条优化都能有完整证明。后面每次放宽边界，都指出原有证明何处需要变化。

## 第二课：从字符串到抽象语法树，先建立安全而明确的语言

抽象语法树通常简称AST。抽象的意思是它保留运算结构，而不必保留所有括号和空格。表达式x+a*b的根是加法，右子树是乘法；表达式(x+a)*b的根是乘法，左子树是加法。若只从左到右看字符，而不处理优先级，就可能把两者当作同一个程序。对x=1、a=2、b=3，正确结果分别为7和9，这个手算例足以抓出最初级的解析错误。

下面使用Python标准库ast完成语法识别，但把Python语法树转换成我们自己的小语言。重要区别是我们不调用eval，也不接受Python全部语义。代码只允许明确列出的节点；像函数调用、属性访问、下标操作、列表构造都会拒绝。这样教学编译器的语言边界由代码直接体现，不依赖读者自觉只输入简单内容。

常量的保存方式需要特别解释。Python浮点对象通常具有比float32更宽的表示，而我们的语言明确要求float32。constant先把数值编码成四个字节，再以十六进制字符串保存。这样后面解码获得的值就是语言中的精确常量，同时保留正零与负零的不同位模式。若直接把Python浮点值用作公共子表达式键，正零和负零比较相等，可能在需要区分符号位的扩展语言里埋下错误。

~~~python
import ast
import math
import struct
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Expr:
    op: str
    args: tuple = ()
    payload: str = ""

def constant(value):
    encoded = struct.pack("<f", float(value))
    rounded = struct.unpack("<f", encoded)[0]
    if not math.isfinite(rounded):
        raise ValueError("教学语言仅接受有限 float32 常量")
    return Expr("const", (), encoded.hex())

def constant_value(node):
    if node.op != "const":
        raise TypeError("不是常量")
    raw = bytes.fromhex(node.payload)
    return np.float32(struct.unpack("<f", raw)[0])

def parse_expression(source, allowed_names=("x", "a", "b")):
    allowed = frozenset(allowed_names)
    operators = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul"}

    def visit(node):
        if isinstance(node, ast.Name):
            if node.id not in allowed:
                raise ValueError("未知输入名: " + node.id)
            return Expr("input", (), node.id)
        if isinstance(node, ast.Constant):
            if type(node.value) not in (int, float):
                raise ValueError("常量必须是数字")
            return constant(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in operators:
            return Expr(operators[type(node.op)],
                        (visit(node.left), visit(node.right)))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return Expr("neg", (visit(node.operand),))
        raise ValueError("不支持的语法节点: " + type(node).__name__)

    parsed = ast.parse(source, mode="eval")
    return visit(parsed.body)

tree = parse_expression("(x*a+b)*(x*a+b)")
assert tree.op == "mul"
assert tree.args[0] == tree.args[1]
for forbidden in ("x[0]", "x.sum()", "__import__('os')", "x / 2"):
    try:
        parse_expression(forbidden)
    except ValueError:
        pass
    else:
        raise AssertionError("非法语法未被拒绝")
assert constant(0.0).payload != constant(-0.0).payload
print(tree)
~~~

dataclass为Expr生成初始化、相等比较等方法，frozen表示创建后不能修改字段。节点不可变有两个用处：一是后面的缓存可以把节点作为键，二是某次优化不会悄悄改变其他地方共享的节点。args是子节点元组，payload则只在输入或常量节点上有意义。二元操作有两个子节点，一元负号只有一个，叶子节点没有子节点。

visit函数递归地把一棵树转成另一棵树。递归不是神秘的新语法：处理父节点前，先调用同一个函数处理每个子节点。遇到Name就得到输入叶子；遇到Constant就完成float32编码；遇到BinOp就查表映射操作名，再保存左右子树。没有命中的情况立即抛异常，所以错误在解析阶段被定位，而不是拖到运行时突然尝试任意Python操作。

本语言允许负号，但ast通常把负数解析为一元负号作用在正常量上。因此-0.0可能先变成neg(const(+0.0))，常量折叠时再生成负零编码。这里没有假设输入文本中的负号已属于数字字面量。理解这一点有助于阅读真实AST：源文件中视觉上连在一起的内容，不一定成为一个节点。

请手工写出x*(2+3)的树：根是mul，左边是input(x)，右边是add(const(2),const(3))。它不是先乘2再加3。再数本章平方表达式的树节点：根乘法一处，两份加法各一处，两份内部乘法各一处，六个输入叶子，总计十一处。后面把相同子树合并成有向无环图时，只剩三个输入节点与三个运算节点，工作量的变化由结构明确体现。

这个解析器并非生产沙箱。非常深或非常长的表达式仍可能消耗过多解析资源，我们的项目入口应限制源码长度和树深度。白名单解决的是语言功能边界，并不自动解决资源限制。学习时只处理几十个节点，后面的小项目会要求加入长度和深度拒绝测试，把正确性与资源约束都写成可观察行为。

## 第三课：解释执行、类型与形状，先定义正确结果

在优化之前，必须先有清楚的原始语义。我们的输入统一转换为NumPy对象后检查dtype：必须恰好是float32，不能悄悄把float64、整数或Python列表混进来。理由不是float64不好，而是不同类型会改变舍入、溢出、存储和生成代码，编译器不能在不知情的情况下改变契约。想支持更多类型，应在类型系统和缓存键里明确增加。

类型不仅是float32这一项。对于张量程序，形状、布局、设备、别名与可变性也可能影响合法操作和代码生成。本节先只推断形状：标量形状是空元组，数组形状如(3,)；两个相同形状相加保持形状；标量与数组相加得到数组形状；其他组合拒绝。NumPy本来支持更广泛广播，但我们刻意不让解释器自动获得未定义的能力。

~~~python
def validate_inputs(env):
    checked = {}
    for name, value in env.items():
        array = np.asarray(value)
        if array.dtype != np.float32:
            raise TypeError(name + " 必须为 float32")
        checked[name] = array
    return checked

def merge_shape(left, right):
    if left == right:
        return left
    if left == ():
        return right
    if right == ():
        return left
    raise ValueError("仅支持同形状或标量广播")

def infer_shape(node, env):
    if node.op == "const":
        return ()
    if node.op == "input":
        if node.payload not in env:
            raise ValueError("缺少输入: " + node.payload)
        return env[node.payload].shape
    shapes = [infer_shape(arg, env) for arg in node.args]
    if node.op == "neg":
        return shapes[0]
    return merge_shape(shapes[0], shapes[1])

def apply_operation(op, values):
    if op == "neg":
        return np.asarray(np.negative(values[0]), dtype=np.float32)
    left, right = values
    if op == "add":
        result = np.add(left, right)
    elif op == "sub":
        result = np.subtract(left, right)
    elif op == "mul":
        result = np.multiply(left, right)
    else:
        raise ValueError("未知操作: " + op)
    return np.asarray(result, dtype=np.float32)

def interpret(node, env):
    checked = validate_inputs(env)
    infer_shape(node, checked)

    def run(current):
        if current.op == "const":
            return np.asarray(constant_value(current), dtype=np.float32)
        if current.op == "input":
            return checked[current.payload]
        return apply_operation(current.op, [run(a) for a in current.args])

    return run(node)

inputs = {
    "x": np.array([1, 2, 3], dtype=np.float32),
    "a": np.array(2, dtype=np.float32),
    "b": np.array(1, dtype=np.float32),
}
expected = np.array([9, 25, 49], dtype=np.float32)
assert infer_shape(tree, validate_inputs(inputs)) == (3,)
np.testing.assert_array_equal(interpret(tree, inputs), expected)
bad_inputs = dict(inputs, a=np.ones(2, dtype=np.float32))
try:
    interpret(tree, bad_inputs)
except ValueError:
    pass
else:
    raise AssertionError("形状检查失效")
~~~

这里check、infer与run的顺序非常重要。先检查类型，再推断所有操作的形状，最后执行数值计算。这样两条不兼容长度的数组不会运算到一半才出现底层广播异常。更复杂的编译器可能在编译期不知道精确形状，此时推断符号关系并在运行时检查条件，但它仍然需要回答“合法性在哪里被证明”。

apply_operation不含任何优化，它按AST规定的顺序调用NumPy操作。对平方例，左子树先算一次x*a+b，右子树又算一次相同结果，最后相乘。我们保留这种重复，因为它提供明确的原始程序。若上来就手工复用中间值，后面无法观察公共子表达式消除到底做了什么，也很难证明优化前后关系。

注意interpret返回输入叶子时可能直接返回输入数组，而非新分配存储。对纯表达式“x”，这个行为符合本章只约束数值、不承诺新输出存储的契约。若框架要求每次输出必须是新张量，包装层就要显式复制，并在fake元信息与alias规则中一致声明。K10的算子集成已经说明，数值相同并不等于完整算子合同相同。

再手算一个形状错误：x形状(2,3)，a形状(3,)，b是标量。NumPy可以把a沿第一维广播，但本语言会拒绝，因为两个非标量形状不相等。这个拒绝是正确执行契约，不是解释器能力退化。以后加入一般广播时，需要同时修改merge_shape、低层索引映射、测试与缓存特化，不能只删掉检查让NumPy替你承担语义。

缺少输入名和未知输入名也应分清。解析时allowed_names决定源码能提到哪些名字；执行时env决定本次调用实际提供了什么。源码合法地引用b，但env没有b，是绑定错误。env多给一个未使用的c，可以选择允许或拒绝，我们的实现允许它；这不会让源码突然能够引用c。编译系统中“语言中的符号”和“某次调用的数据绑定”正是两个阶段。

至此你已经拥有完整参考执行器。后面任何优化都要对照它，不应拿优化后的版本自己生成期望值再验证自己。对固定小输入保存手算结果，对随机输入比较独立表示的执行结果，对边界值特别检查符号零、溢出和非有限数。编译器正确性通常不是靠“程序没崩”证明，而是靠语义契约、变换条件和差分测试共同支撑。

## 第四课：常量折叠为何简单，代数化简为何可能危险

常量折叠指在编译期间计算完全由常量组成的子表达式。例如x*(2+3)的右子树不依赖输入，可以提前算成float32常量5。运行时乘法仍然存在，但每次调用不必再加2和3。关键条件是使用目标语言的数值语义折叠：如果语言规定float32每步舍入，就不能先用无限精度或float64把整式算完，再在最后转一次float32。

下面的fold_constants先递归折叠子节点，再判断当前节点的所有参数是否都是常量。若是，就使用同一个apply_operation，保证操作次序和float32结果与解释器一致。若计算产生无穷或NaN，则保留原表达式，因为我们的常量编码契约只接受有限值；这不表示运行时不允许出现无穷，而是这个有限常量IR没有对应字面量表示。

~~~python
def fold_constants(node):
    if node.op in ("const", "input"):
        return node
    new_args = tuple(fold_constants(arg) for arg in node.args)
    if all(arg.op == "const" for arg in new_args):
        values = [np.asarray(constant_value(a)) for a in new_args]
        with np.errstate(over="ignore", invalid="ignore"):
            value = apply_operation(node.op, values)
        if np.isfinite(value).item():
            return constant(value.item())
    return Expr(node.op, new_args, node.payload)

fold_example = parse_expression("x*(2+3)")
folded = fold_constants(fold_example)
assert folded.args[1] == constant(5)
np.testing.assert_array_equal(
    interpret(fold_example, inputs), interpret(folded, inputs)
)

signed_zero = fold_constants(parse_expression("-0.0"))
assert np.signbit(constant_value(signed_zero))
nan_input = np.array([np.nan, 2], dtype=np.float32)
with np.errstate(invalid="ignore"):
    actual_zero_product = np.multiply(nan_input, np.float32(0))
assert np.isnan(actual_zero_product[0])
assert not np.array_equal(actual_zero_product, np.zeros(2, dtype=np.float32))
~~~

为什么不顺手把x*0改成0？因为x可能是NaN或无穷，乘零结果不总是零。为什么不无条件把x+0改成x？在严格IEEE语义中，符号零与舍入模式可能使某些观察不同。为什么不把(a+b)+c变成a+(b+c)？float32加法不满足实数世界的结合律。优化器必须知道语言允许哪些等价关系，不能只凭中学代数推出程序变换。

考虑float32的a=100000000、b=-100000000、c=1。先加a+b得到0，再加c得到1；先加b+c时，那个1可能因精度不足被舍去，结果仍为-100000000，再加a得到0。这里两个程序没有越界、没有线程竞态，仅改变括号就得到不同结果。归约树、Split-K与编译器重新结合都可能涉及同类问题，所以数值容差必须建立在明确契约上。

融合乘加FMA计算a*b+c时通常只在最终结果舍入一次，而分开的乘法与加法可能各舍入一次。FMA往往提高局部精度，但这仍然是行为变化，不能承诺逐比特一致。下面用能被float32精确表示的数，构造清晰例子：a=1+2^-23，b=1-2^-23，c=-1。精确乘积是1-2^-46；先舍入乘积为1，再减1得到0；融合计算可以保留-2^-46。

~~~python
fma_a = np.float32(1 + 2**-23)
fma_b = np.float32(1 - 2**-23)
fma_c = np.float32(-1)
separate = np.float32(np.float32(fma_a * fma_b) + fma_c)
# 这三个特定值的精确乘加可由 binary64 表示，故可模拟单次最终舍入。
single_round = np.float32(float(fma_a) * float(fma_b) + float(fma_c))
assert separate == np.float32(0)
assert single_round == np.float32(-(2**-46))
assert separate != single_round
~~~

注释特意限制了这个模拟的适用范围：一般float32乘加的精确结果是否都可用某段float64代码可靠模拟，需要更完整分析，不能把这个例子扩张为通用FMA仿真器。这里选择的整数幂结构使计算可手算且能被所用表示保存。你应写出精确代数，再看每一步的舍入位置，而不是只看打印的小数相近就判定等价。

fast math通常是一组放宽数值约束的选项或假设，可能允许不同的重新结合、近似数学函数、特殊值处理或融合策略。不同工具的选项集合不完全相同，不能看到“fast”就理解为免费加速。正确流程是先决定模型和算子的误差预算，再查看实际编译选项，最后验证极端输入与端到端影响。任何性能报告都应保留影响数值语义的选项。

常量折叠的另一个限制是副作用。假设某函数表面返回常量，却同时写日志、更新随机数状态或检查异常，删除调用会改变程序行为。我们的语言只有纯算术，所以折叠条件简单；扩展到load、store、随机数和通信时，必须给操作增加效果信息，或者使用显式状态依赖。下一课会用内存写入举例说明“相同代码”为什么不等于“相同值”。

## 第五课：公共子表达式、SSA与死代码，编译器开始改变工作结构

公共子表达式消除常简称CSE。它寻找多处相同且可安全复用的计算，把它们变成一次计算、多处使用。平方例中两份x*a+b完全相同，输入在整个表达式执行期间不变，操作没有副作用，所以可以共用结果。树变成图以后，一个节点可以有多个使用者；图仍然没有环，因为本章没有循环和递归调用。

静态单赋值形式简称SSA，核心条件是每个逻辑值名字只被定义一次。t0、t1、t2各产生一个新值，不在后面被重新赋值。SSA不是说程序只能写一次内存，也不是说物理寄存器永远不能复用；它约束的是中间表示中值的命名。这样分析“这个使用来自哪个定义”变得直接，也方便计算存活范围和删除无用计算。

下面把Expr转换成指令列表。每条Instruction保存操作名、参数值编号和载荷。emit先处理依赖再添加当前指令，因此列表天然按拓扑顺序排列：任何输入编号都小于使用它的指令编号。memo以不可变Expr为键，第二次遇到同一子树时直接返回旧编号。输入节点和常量节点也可以共用，因此树中的重复叶子不会反复建立。

~~~python
@dataclass(frozen=True)
class Instruction:
    op: str
    inputs: tuple = ()
    payload: str = ""

@dataclass(frozen=True)
class Program:
    code: tuple
    result: int

def lower_to_ssa(root):
    code = []
    memo = {}

    def emit(node):
        if node in memo:
            return memo[node]
        arguments = tuple(emit(arg) for arg in node.args)
        index = len(code)
        code.append(Instruction(node.op, arguments, node.payload))
        memo[node] = index
        return index

    result = emit(root)
    return Program(tuple(code), result)

def execute_ssa(program, env):
    checked = validate_inputs(env)
    values = []
    for index, instruction in enumerate(program.code):
        if any(j >= index or j < 0 for j in instruction.inputs):
            raise ValueError("不是拓扑有序 SSA")
        if instruction.op == "input":
            value = checked[instruction.payload]
        elif instruction.op == "const":
            value = np.asarray(constant_value(
                Expr("const", (), instruction.payload)))
        else:
            arguments = [values[j] for j in instruction.inputs]
            shapes = [a.shape for a in arguments]
            if len(shapes) == 2:
                merge_shape(shapes[0], shapes[1])
            value = apply_operation(instruction.op, arguments)
        values.append(value)
    if not 0 <= program.result < len(values):
        raise ValueError("非法返回编号")
    return values[program.result]

program = lower_to_ssa(fold_constants(tree))
assert len(program.code) == 6
assert sum(i.op == "mul" for i in program.code) == 2
np.testing.assert_array_equal(execute_ssa(program, inputs), expected)
for index, instruction in enumerate(program.code):
    print(index, instruction)
~~~

实际编号依次是0=x，1=a，2=mul(0,1)，3=b，4=add(2,3)，5=mul(4,4)。这里同一个值4在最终乘法参数里出现两次。值4只有一次定义，并不限制使用次数。原树有两次内部乘法、两次加法和一次外部乘法，总共五次算术；SSA只有两次乘法和一次加法，总共三次算术。对长度三的数组，元素级算术从十五次减到九次。

死代码消除简称DCE，删除不会影响返回值或其他可观察效果的指令。下面人为在末尾加入一条常量指令，但返回编号仍指向平方结果。由于本语言没有副作用，只有从返回值逆向可达的节点需要保留。真正编译器还要保留外部写入、可能观察到的异常、同步或其他效果，不能把“返回值不使用”直接当作通用删除条件。

~~~python
def eliminate_dead_code(program):
    needed = set()

    def mark(index):
        if index in needed:
            return
        if not 0 <= index < len(program.code):
            raise ValueError("引用越界")
        needed.add(index)
        for parent in program.code[index].inputs:
            mark(parent)

    mark(program.result)
    remap = {}
    new_code = []
    for old_index, instruction in enumerate(program.code):
        if old_index not in needed:
            continue
        new_index = len(new_code)
        remap[old_index] = new_index
        new_code.append(Instruction(
            instruction.op,
            tuple(remap[parent] for parent in instruction.inputs),
            instruction.payload,
        ))
    return Program(tuple(new_code), remap[program.result])

extra = Instruction("const", (), constant(999).payload)
with_dead = Program(program.code + (extra,), program.result)
clean = eliminate_dead_code(with_dead)
assert clean == program
np.testing.assert_array_equal(execute_ssa(clean, inputs), expected)
~~~

mark的递归沿输入边逆向遍历；needed集合避免对共享节点重复处理。第二个循环保持原拓扑顺序，并建立旧编号到新编号映射。先添加依赖，后添加使用者，因此生成新参数编号时，remap里一定已有所需条目。若输入列表不是拓扑有序，应该先验证或重新拓扑排序，不能假设任意手写Program都满足条件。我们的生成器负责保证这个前置条件。

公共子表达式消除最常见的错误发生在内存操作。设第一次load(p)读到3，随后store(p,7)，再执行相同文本的load(p)，第二次应得到7。若只按“操作名与指针相同”缓存，第二次会错误复用3。需要内存版本、效果链、别名分析或其他机制证明两次读取之间没有可能改变该位置的写入。随机数调用同样不能因为函数名和参数相同就合并。

再考虑分支。变量v在if两边分别赋值，合流处使用v，SSA通常用合流选择节点表达“值来自哪条控制边”；LLVM里常见的是phi。它不是把两个分支的结果相加，也不是一定执行两个分支。当前直线表达式没有控制流，因此无需实现phi；但你阅读真实IR时应把控制依赖和数据依赖分开，不能用本章直线列表去假装已经解释全部编译器。

## 第六课：存活范围、寄存器分配与spill，值编号不等于物理寄存器

SSA可以为每次计算创建新名字，真实芯片却只有有限寄存器。寄存器分配要让同时需要的值不占用同一位置，同时尽可能重用已经死亡的值所占空间。所谓值存活，是它已经产生、未来仍会被使用。产生但再也不会使用的值可以被删除或释放；已经用完的值不必一直留到函数结束。

仍看六条指令：值0和1在指令2使用后，若后面没有其他使用，就不再存活；值2要留到指令4；值3也在指令4最后使用；值4在指令5被使用两次，但这两个使用属于同一操作。可以把值0与后来值3安排到同一物理位置，只要在覆盖前确认值0已经死亡。源码中的不同变量名不会强制获得不同物理寄存器。

下面只做保守的存活区间分析，不尝试模拟GPU寄存器分配。区间从定义编号开始，到最后使用编号结束；返回值延伸到整个程序结束。把使用操作的输入和输出同时计为活跃，得到一个保守峰值，因为某些真实指令可以复用已用完输入的位置。这个模型适合比较变换前后的压力趋势，却不能直接预测实际register count。

~~~python
def live_intervals(program):
    last_use = list(range(len(program.code)))
    for user, instruction in enumerate(program.code):
        for producer in instruction.inputs:
            last_use[producer] = max(last_use[producer], user)
    last_use[program.result] = len(program.code)
    return [(i, last_use[i]) for i in range(len(program.code))]

def live_profile(program):
    intervals = live_intervals(program)
    return [
        [i for i, (start, end) in enumerate(intervals) if start <= step <= end]
        for step in range(len(program.code))
    ]

intervals = live_intervals(program)
profile = live_profile(program)
assert intervals == [(0, 2), (1, 2), (2, 4), (3, 4), (4, 5), (5, 6)]
assert max(map(len, profile)) == 3
print(intervals, profile)
~~~

为什么CSE有时反而增加寄存器压力？假设一个便宜表达式在程序开头和末尾各用一次。重复计算时，中间值可以很快死亡；合并后，它必须从开头一直保存到末尾。节省一次算术可能换来更长存活区间，甚至触发spill。编译器有时选择重计算便宜值，就是为了降低存储压力。算术次数少不是唯一目标，存储位置、并行度和依赖长度都影响成本。

spill指无法一直放在寄存器中的值被存到其他存储位置，需要时再读回来。CUDA中线程私有的local memory是地址空间概念，并不等于物理上就在SM旁边的一块高速私有存储；相关访问可能经过缓存并最终到设备内存。看到源码没有global数组访问，仍可能在产物里看到spill读写。K04解释的存储层次在这里重新出现了。

编译器还需要指令选择与指令调度。指令选择把抽象操作映射为目标支持的指令组合；调度在保持依赖和内存语义的前提下安排先后，尝试隐藏延迟或提高吞吐。某个源代码表达式可能对应多条机器指令，多个表达式也可能融合成一条。不要逐行强行对齐源码和机器码，应先识别数据流：哪个地址被加载、哪组值参与运算、结果写到哪里。

循环展开会把多次迭代的操作放进同一个较大区域，提供更多调度机会，也可能扩大代码、延长存活范围。Tile变大能增加复用，但也增加累加器数量。一个线程保存8×8个float32累加器，光逻辑累加器就有64个值，尚未计算地址、输入片段和临时量。实际寄存器数由编译结果决定，但这个估算已经能解释为什么更大Tile不必然更快。

屏障和异步操作进一步限制调度。编译器不能为了局部速度把一个必须在屏障后读取的共享值随意移到屏障前。异步拷贝的发起不等于完成，相关等待关系也是语义的一部分。普通表达式的纯数据依赖分析不够覆盖这些操作，需要控制、内存效果和同步协议。K17的代际与stage释放逻辑正是不能被天真重排破坏的约束。

## 第七课：真正编译一个C++程序，区分预处理、编译、汇编与链接

现在把概念映射到实际工具。下面程序实现y=2*x+1，并打印总和。使用数组而非动态分配，是为了把注意力集中在编译产物；函数接受指针与长度，仍需调用者提供足够存储。extern "C"指定C语言链接方式，使导出符号更容易在汇编里定位；它没有把函数内部语法变成C，也不改变浮点乘加的数学含义。

~~~cpp
#include <cstddef>
#include <iostream>

extern "C" void affine(const float* x, float* y, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        y[i] = 2.0f * x[i] + 1.0f;
    }
}

int main() {
    const float x[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float y[5] = {};
    affine(x, y, 5);
    float sum = 0.0f;
    for (float value : y) {
        sum += value;
    }
    if (sum != 35.0f) {
        return 1;
    }
    std::cout << sum << '\n';
    return 0;
}
~~~

把它保存为临时学习目录中的affine.cpp，然后运行以下命令。命令里的build-k15是自己的实验输出目录，不需要删除仓库中其他文件。第一次执行mkdir建立目录，后面每种产物都用不同后缀保存。请先理解每一步作用，再一次性运行；遇到失败时记录最早出现的错误，而不是只看最后一行“退出非零”。

~~~bash
mkdir -p build-k15
g++ -std=c++17 -E affine.cpp -o build-k15/affine.ii
g++ -std=c++17 -O0 -S affine.cpp -o build-k15/affine-O0.s
g++ -std=c++17 -O2 -S affine.cpp -o build-k15/affine-O2.s
g++ -std=c++17 -O2 -c affine.cpp -o build-k15/affine.o
g++ build-k15/affine.o -o build-k15/affine
./build-k15/affine
~~~

预处理-E展开头文件、宏与条件编译，结果仍是C++源形式。包含iostream会带来大量标准库声明，因此.ii很长，不代表你的算法突然复杂。编译到汇编-S完成语言分析、优化和目标生成，输出面向当前主机架构的汇编文本。.s不是GPU PTX，也不是可直接运行的完整程序。生成目标文件-c会得到包含机器码、符号与重定位信息的文件，但仍可能依赖其他定义。

链接把目标文件和所需库组合起来，解析跨文件符号，处理重定位，生成可执行文件或共享库。缺少函数定义通常表现为链接错误；语法缺分号通常更早在编译阶段出错。二者都可能出现在同一条g++命令里，因为g++是会驱动多个步骤的工具，但职责仍然不同。想定位问题，临时拆开-c与链接步骤往往比猜测更有效。

这里O0和O2都应该打印35，但汇编结构可能明显不同。优化构建可能消除某些循环、把main中的已知输入提前计算，或采用向量指令；具体取决于编译器版本、目标架构和选项。教学验收不是必须找到某个预定汇编助记符，而是找到affine符号附近的边界、地址步进和运算，再解释main是否还实际调用它。

如果用于基准，固定输入且结果只用于一个编译期可知检查，编译器可能提前求出答案。打印结果使结果可观察，但不保证原算法必须在运行时逐项执行，因为编译器也能打印提前算出的35。真正测量应使用运行时输入，并检查产物或使用恰当的基准框架。把volatile加到所有数组上会改变内存语义和优化空间，通常不是理解死代码删除的理想替代。

你可以改变n为来自命令行的有限整数，再填充数组，观察编译器无法提前知道长度时的变化。注意先验证输入范围，避免把负数错误转换成巨大无符号长度。这个扩展连接K02的类型转换与K14的测量合同：性能实验本身也是程序，需要正确的数据来源、边界和结果观察方式。

## 第八课：CUDA源文件为什么会走两条编译路径

CUDA C++源文件通常同时包含主机代码和设备代码。主机代码在CPU上准备数据、分配资源、发起调用；设备代码在GPU上执行。nvcc是编译驱动工具，会协调主机编译器和设备编译阶段，而不是一个独自完成所有工作、也不依赖其他编译器的单体。某些报错来自主机C++前端，某些来自设备编译或汇编，读日志时应先识别来源。

先准备一个只含设备内核的完整小文件add_kernel.cu。它不包含main，因此可以生成设备产物，但不能直接链接成普通可执行程序。每个线程计算一个i，对边界内元素完成加法。无符号线程坐标先转换到size_t再相乘，避免在更窄的中间类型里溢出；这种类型选择会影响地址计算指令，不只是代码风格。

~~~cpp
#include <cstddef>

extern "C" __global__
void add_kernel(const float* x, const float* y, float* out, std::size_t n) {
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x
                  + threadIdx.x;
    if (i < n) {
        out[i] = x[i] + y[i];
    }
}
~~~

下面是目标设备为H100时可用于阅读产物的命令示例。compute_90表示生成对应虚拟目标的PTX路径，sm_90表示具体本机目标；它们不是任意字符串替换游戏。工具链必须支持这些目标，设备也必须满足产物要求。RTX5070Ti使用自己的能力与支持路径，不能把本节H100命令的成功当作它已经通过验证。目标选择的硬件依据在K16展开。

~~~bash
nvcc -std=c++17 -O2 -lineinfo --ptx -arch=compute_90 add_kernel.cu -o build-k15/add.ptx
nvcc -std=c++17 -O2 -lineinfo --cubin -arch=sm_90 add_kernel.cu -o build-k15/add.cubin
nvdisasm build-k15/add.cubin
~~~

这里还要辨别工具输入：cuobjdump用于CUDA二进制产物的检查，不应把已经是文本的PTX文件当作必须再次反汇编的二进制。实际阅读add.ptx直接用编辑器或文本命令；下面给出文本阅读与二进制反汇编的对应选择。若希望读取二进制里嵌入的PTX，则对包含相应代码的fatbinary或主机二进制使用cuobjdump，并确认构建时确实保留了PTX镜像。

~~~bash
sed -n '1,160p' build-k15/add.ptx
cuobjdump --dump-sass build-k15/add.cubin
nvdisasm build-k15/add.cubin
~~~

PTX是虚拟指令集表示，包含目标、参数、寄存器声明和指令。cubin是设备二进制，通常采用ELF格式，携带针对具体架构的代码和相关元信息。SASS常用于称呼NVIDIA目标机器指令的反汇编表示。GPU执行的是适合设备的本机代码，不是每次运行都让硬件逐行解释PTX文本。PTX到本机指令的映射由工具链与目标共同决定。

fatbinary可以包含多种设备代码镜像，例如某些目标的cubin和一份可用的PTX。主机程序中可能嵌入这种设备产物；运行时和驱动根据设备与兼容条件选择可加载代码，必要时由驱动JIT把支持的PTX编译为本机代码。保留PTX可能提供某些向前兼容路径，但不是保证任意旧驱动、任意新指令和任意架构专用目标都能互通。

真实带主机包装的程序可以用-gencode明确同时包含目标cubin和PTX。例如K03的完整CUDA主机程序若另存为add_full.cu，可用以下构建形式。这里的文件必须确实含main与错误检查，不能把只含内核的add_kernel.cu误当同一个输入。读取产物和实际启动是不同验证等级；生成cubin并不证明内核数值正确，也不证明目标GPU能成功执行所有路径。

~~~bash
nvcc -std=c++17 -O2 -lineinfo   -gencode arch=compute_90,code=sm_90   -gencode arch=compute_90,code=compute_90   add_full.cu -o build-k15/add_full
cuobjdump --dump-ptx build-k15/add_full
cuobjdump --dump-sass build-k15/add_full
~~~

若驱动不能理解生成PTX使用的版本，可能在装载或JIT阶段失败；若二进制没有适用于设备的镜像，也会出现对应装载错误。这里“Toolkit版本”“驱动能力”“PTX版本”“目标架构”“实际GPU”是不同轴。nvidia-smi显示的CUDA能力不等于本地nvcc已经安装的版本，PyTorch打包的运行组件也不等于你的系统Toolkit。收集完整版本清单才能定位真实不匹配。

## 第九课：读PTX与LLVM IR，先追值和地址，再追助记符

读中间表示的第一遍，不应追求记住所有指令。先找函数入口、参数、基本块、分支、内存读写和返回。基本块是中间没有跳入跳出、通常按顺序执行的一组指令；分支连接基本块，形成控制流图。源码中的if往往变成条件判断与跳转，也可能在后续阶段变成谓词执行，具体取决于目标和优化选择。

下面是对应一维加法逻辑的PTX教学摘录，不是本环境生成的产物，也不是可独立汇编的完整文件。为缩短轨迹，它假定元素索引能够放进无符号32位整数；第八课完整CUDA源码使用size_t，因此真实产物可能使用不同宽度和指令组合。参数声明、参数加载、地址空间转换、寄存器声明与部分地址计算在摘录中省略，阅读时不能把省略当作这些步骤不存在。

~~~text
mov.u32      %r_tid,  %tid.x;
mov.u32      %r_bid,  %ctaid.x;
mov.u32      %r_bdim, %ntid.x;
mad.lo.u32   %r_i, %r_bid, %r_bdim, %r_tid;
setp.ge.u32  %p_out, %r_i, %r_n;
@%p_out bra  DONE;
mul.wide.u32 %rd_offset, %r_i, 4;
add.u64     %rd_xi, %rd_x, %rd_offset;
ld.global.f32 %f_x, [%rd_xi];
ld.global.f32 %f_y, [%rd_yi];
add.f32     %f_out, %f_x, %f_y;
st.global.f32 [%rd_oi], %f_out;
DONE:
ret;
~~~

mov从特殊寄存器读取线程号、块号和块大小；mad.lo表示整数乘加并保留相应低位结果；setp把比较结果写入谓词；带@的分支在谓词满足时执行。mul.wide把32位索引乘4得到更宽字节偏移，随后与64位基地址相加。f32表示单精度浮点操作，global说明所访问地址属于全局存储空间。助记符的后缀是语义的一部分，不能读成装饰。

手算blockIdx.x=2、blockDim.x=4、threadIdx.x=1，得到元素索引9。若n=10，这个线程有效；float32元素宽4字节，偏移是36。若x基地址在教学模型里为4096，读地址就是4132。若threadIdx.x改成2，索引变10，谓词为真，线程跳到结束，不应再执行越界load。这个轨迹把K01地址与K03边界直接连接到指令层。

特别区分元素数量和字节数量。若把n=10误传为40，条件会允许本不存在的元素；若把offset=i误当成字节偏移，线程读到未对齐且错误位置。高层指针算术x+i会自动乘元素宽度，低层整数地址计算通常需要显式体现。读PTX时应给每个寄存器旁写单位：元素、字节、布尔或浮点值。单位检查能在不懂全部ISA的情况下抓住许多问题。

LLVM IR则是另一种中间表示，并不专属于GPU。下面完整的小函数表达标量仿射操作。float是值类型，%x是参数，%scaled和%shifted是SSA值，entry是基本块标签，ret返回结果。没有任何load或store，因为输入已经作为标量值传入；若源程序处理指针数组，IR还要出现地址计算与内存操作。

~~~llvm
define float @affine_scalar(float %x) {
entry:
  %scaled = fmul float %x, 2.000000e+00
  %shifted = fadd float %scaled, 1.000000e+00
  ret float %shifted
}
~~~

这段代码只用于阅读语义，本章没有声称已通过本机LLVM工具验证。实际工具可能附加目标三元组、数据布局、属性、调试元信息与优化标志。不要把最简函数当成真实项目输出的固定模板。目标数据布局影响指针宽度和类型对齐，目标特性影响指令选择；相同IR文本在不同目标上也可能产生不同机器码。

LLVM的浮点指令可以携带特定fast-math标志，整数运算也可能携带关于溢出的假设。阅读这些标志时应查官方语言参考，理解违反假设的后果，而不是只看算术操作名字。某些IR语义包含poison等中间状态，它们不是普通的随机数或自动抛出的Python异常。初学阶段先不主动生成这类强假设，等能清楚证明边界后再引入。

最后把PTX虚拟寄存器和SASS物理寄存器分开。PTX里声明许多临时寄存器，不表示实际每线程必然占用同样数量；后端可能合并、删除、复用或spill。实际资源报告和反汇编才接近目标分配结果。反过来，只看到某个物理寄存器编号，也不能反推出它对应哪一个源代码变量。正确阅读单位是数据依赖与存储行为，而非变量名称的一一配对。

## 第十课：ABI、链接与别名，源代码正确仍可能在边界处出错

应用二进制接口简称ABI，规定编译后不同组件如何交换数据和调用函数。它涉及参数类型、宽度、对齐、返回方式、符号命名与调用约定。你可以把它理解成两段机器代码之间的合同。Python层写一个整数、C++层接一个size_t、设备参数表读取一个64位值，这几步必须一致；表面都叫n不代表字节布局自动吻合。

假设主机按32位整数打包n，而设备按64位参数读取。设备可能把后面参数的一部分也当成n的高位，得到巨大长度。此时内核内部i<n判断本身完全正确，却保护不了错误参数。类似地，结构体成员之间可能有对齐填充，不能把各字段大小简单相加后就断言结构体大小。K02的sizeof、alignof、标准布局与生命周期知识在这里直接成为运行正确性的条件。

CUDA内核参数中的指针是设备可访问地址，不是普通主机数组地址的通用替代。不同内存分配方式与统一地址空间机制有各自规则，不能因为两个值都打印成十六进制就认定GPU可访问。学习阶段使用框架设备张量或明确cudaMalloc等方式获得的存储，通过规范接口传递，避免自己随意拼接指针整数。地址看起来合法不等于对象生命周期和访问权限合法。

符号是链接器用来找到定义的名字。C++为支持重载等语言功能，常把函数签名编码进链接符号；extern "C"可以在适用场合使用C链接约定，方便跨语言接口，但不会自动解决参数宽度、数组布局与异常传播。若头文件声明和实现定义不一致，可能在编译或链接阶段报错，也可能在不安全的外部接口中变成更难定位的运行错误。

跨文件编译要求声明与定义分工清晰。声明告诉当前编译单元某个函数的类型；定义提供函数体。只包含声明可以生成对外部符号的引用，最终链接必须找到兼容定义。CUDA还可能涉及设备代码的独立编译与设备链接，不能把普通主机链接成功当作所有设备符号都已解决。使用可重定位设备代码时，应按当前nvcc文档理解编译与链接选项，而不是机械复制某个旧构建命令。

别名分析则回答两个指针是否可能访问同一存储。考虑y[i]=x[i]+1，若x与y完全相同，逐元素原地操作可能合法；若y指向x向后偏移一个元素的位置，前一次写入可能改变后一次读取。编译器能否重排、向量化或并行执行，取决于语言语义和已知别名关系。一个只在不重叠输入上正确的内核，必须把不重叠写入契约，不能默默接受任意视图。

某些编译器扩展的restrict类限定表达程序员承诺的非别名条件。它不是运行时检查，也不会自动复制重叠数组。错误承诺可能让优化器合法地做出与你实际输入不匹配的变换。学习时先用普通指针验证，再讨论在什么范围内能证明不重叠；特别是张量slice、广播与as_strided式视图，形状不同不代表存储不同。

下面用NumPy显式模拟“前向逐项读写”和“先读完再写”的区别，避免只抽象谈论别名。初始数组[1,2,3,4]，输出指向后三个元素，输入指向前三个元素。逐项加十会让新值继续传播，得到[1,11,21,31]；先保存旧输入再写会得到[1,11,12,13]。二者都没有越界，差异完全来自依赖与重叠。

~~~python
sequential_alias = np.array([1, 2, 3, 4], dtype=np.float32)
for i in range(3):
    sequential_alias[i + 1] = sequential_alias[i] + np.float32(10)

snapshot_alias = np.array([1, 2, 3, 4], dtype=np.float32)
old_input = snapshot_alias[:3].copy()
snapshot_alias[1:] = old_input + np.float32(10)

np.testing.assert_array_equal(sequential_alias, [1, 11, 21, 31])
np.testing.assert_array_equal(snapshot_alias, [1, 11, 12, 13])
assert not np.array_equal(sequential_alias, snapshot_alias)
~~~

这个例子也解释了为什么融合、向量化和多线程化必须遵守输入输出合同。若原程序允许重叠并定义了顺序行为，变换不能随意改成快照行为；若合同明确要求不重叠，则可在该前提下优化。正确性测试应包含允许的别名情况，以及对禁止情况的拒绝检查。仅用彼此独立的新数组测试，可能完全遗漏边界错误。

## 第十一课：Triton与框架编译，各层IR保留了不同的信息

回到Triton。Python包装层准备输入、检查形状类型、计算grid，并选择BLOCK等配置；JIT内核描述一个Program处理的块张量操作。前端首先需要理解这些操作及其类型，再逐步确定布局、线程分工、内存访问和目标指令。块张量里的一个逻辑元素不等于一个固定线程，编译器需要把抽象并行形状映射到真实线程与寄存器组织。

在常见Triton路径中，你可能看到TTIR、面向GPU布局的IR、LLVM IR、PTX与cubin等表示。具体名称、顺序和调试接口会随版本与后端变化，课程不把它们写成所有设备都必须完全相同的流水线。重要的是问每一级还保留什么：高层表示容易看到张量维度和归约；较低层布局表示说明值如何分布；更低层出现指针、寄存器和目标操作。

例如一行softmax在高层是load、max、减法、exp、sum、除法、store。布局降低后，需要决定哪部分元素由哪个线程持有，归约哪些阶段在线程内部完成、哪些需要线程间交换。PTX层可能出现多条shuffle或共享内存相关指令，SASS层又受具体目标影响。你不能从高层只有一个sum就推断机器只有一条“求和所有元素”指令。

constexpr使某些值在编译时确定，例如BLOCK=128。这样arange长度、静态循环次数与部分mask结构可以提前分析。运行时n=100仍然作为参数参与边界判断；把n也特化成常量可能生成更多版本，但不保证收益覆盖编译成本和缓存占用。哪些量特化是工程选择，需要结合输入分布、代码大小和稳定性，而非“越静态越高级”。

框架图编译处理的范围通常比单个内核大。它可能捕获一段张量操作，建立图，检查guard，融合某些操作，调用库实现或生成新内核。一个图不必等于一个内核，一次Python函数调用也可能包含多个图段。graph break说明当前捕获路径在此断开，常见后果是回到其他执行方式或分段执行；它本身不等于模型数值错误。

guard是重用编译产物前检查的条件。假设某版本只支持连续float32、最后一维128，运行时就应核对这些条件，失败时选择其他版本、重新编译或使用回退实现。没有guard而复用错误特化版本，可能导致越界或错误结果；过度严格的guard又会导致频繁重新编译。理解guard是阅读框架编译缓存行为的关键，不应只把重编译看成随机变慢。

动态形状支持也有多种含义。某些形状可作为运行参数，在一个内核中用mask处理；某些形状关系可以用符号表达；某些变化仍需新版本。声称“支持动态”时，应具体写出哪些维度可以变化、范围多大、是否导致新编译、是否需要padding、性能是否稳定。K13的图重放地址稳定性与这里的编译特化是相关但不同的约束，不能互相替代。

源码研读时，先使用本卷固定Triton提交的教程内核建立输入输出和布局图，再跟踪实际安装版本的编译入口与产物。没有读取过的内部函数名不要凭印象写进报告；内部API变化很快，外部教程能运行也不证明某条私有调试属性仍存在。最可靠的记录包含源码提交、安装版本、实际命令、产物片段和未解释问题。

## 第十二课：实现一个可检查的JIT缓存，理解特化键和冷启动

JIT意味着编译发生在程序运行期间，但编译完成的产物仍然可以缓存并重复使用。缓存键必须包含决定产物语义与目标约束的信息。源代码改变、类型改变、布局假设改变、目标改变、编译选项改变，都可能需要新产物。普通输入数值通常不应进入键，否则每换一批数据就重新编译；除非该数值本来被明确声明为编译期常量。

下面做一个教学缓存，编译动作是解析、常量折叠和SSA降低，执行仍由NumPy完成。键包含源码、输入名称、dtype、shape、字节stride、编译器标识、目标标识和配置。这里保守地把所有形状与步长都特化，会产生较多版本；真实系统可用guard和符号条件更灵活地复用，但要有相应正确性证明。这个实现不宣称等同Triton内部缓存格式。

~~~python
import hashlib
import json

def compilation_key(source, env, target, config):
    checked = validate_inputs(env)
    signatures = [
        {
            "name": name,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "strides_bytes": list(array.strides),
        }
        for name, array in sorted(checked.items())
    ]
    record = {
        "source": source,
        "inputs": signatures,
        "compiler": "teaching-expression-v1",
        "target": target,
        "config": config,
    }
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

class TinyJIT:
    def __init__(self):
        self.cache = {}
        self.compilations = 0

    def run(self, source, env, target="numpy-cpu", config=None):
        config = {} if config is None else dict(config)
        key = compilation_key(source, env, target, config)
        if key not in self.cache:
            root = parse_expression(source, allowed_names=tuple(env))
            infer_shape(root, validate_inputs(env))
            optimized = fold_constants(root)
            self.cache[key] = lower_to_ssa(optimized)
            self.compilations += 1
        return execute_ssa(self.cache[key], env)

jit = TinyJIT()
source = "(x*a+b)*(x*a+b)"
np.testing.assert_array_equal(jit.run(source, inputs), expected)
changed_values = dict(inputs, x=np.array([4, 5, 6], dtype=np.float32))
np.testing.assert_array_equal(
    jit.run(source, changed_values), np.array([81, 121, 169], dtype=np.float32)
)
assert jit.compilations == 1
reversed_view = dict(inputs, x=inputs["x"][::-1])
jit.run(source, reversed_view)
assert jit.compilations == 2
jit.run(source, inputs, config={"tile": 128})
assert jit.compilations == 3
~~~

第一次调用把Program放入cache，compilations变1。第二次输入值改变，但形状、类型、步长不变，因此命中旧产物；执行器读取本次env，所以输出随数据变化。第三次x是负步长视图，键不同，保守地产生新版本。第四次配置增加tile=128，虽然教学执行器没有真实tile映射，我们仍把配置视为影响产物的参数，演示配置变更必须进入缓存策略。

为什么不用Python内置hash作为持久文件名？某些对象的哈希受进程随机化等因素影响，不适合作为跨进程稳定内容标识；此外需要规范化序列化，避免字典插入顺序改变表示。这里用排序JSON和SHA256得到稳定内容摘要。摘要只解决标识，不自动证明缓存安全：还需要版本、依赖、文件写入完整性、并发访问和失效策略。课程不把一个字典夸大成生产缓存。

源码依赖也必须考虑。真实内核可能调用辅助函数，读取编译期全局常量，依赖外部库或目标特性。只给最外层函数字符串算摘要，辅助函数变了却键不变，会复用旧产物。我们的语言没有函数调用和外部依赖，因此source足以描述程序逻辑；一旦扩展语言，就必须同步扩展依赖图与键。这是缩小语言边界带来的可证明性。

冷启动成本应拆成导入、初始化、编译、装载、分配、预热和执行，不要把首次调用全部归为编译。热路径也不保证零主机开销，它仍可能检查guard、查缓存、准备参数和提交内核。计时应按K14的方法明确范围，记录第一次与稳定重复样本，避免用一个总时间猜测哪个阶段变慢。缓存命中次数比仅看时间更直接，因为运行时抖动可能掩盖差异。

缓存还需要容量策略。若每个输入长度都产生一个版本，长时间服务可能积累大量产物。可以按shape bucket复用，也可以限制版本数量并淘汰，但任何桶化都必须保证mask、资源和输出语义正确。项目阶段先使用三个固定shape，观察版本数量，再加入有界动态范围；不要一开始自动搜索几百种配置，占用公司GPU来弥补对缓存结构的不理解。

## 第十三课：用三个可复现故障，学会定位最早失效的合同

第一个故障是把数学恒等式当成无条件程序等价。错误优化器看到乘零就返回零数组，普通有限输入全部通过，但NaN输入暴露差异。修复不是把测试中的NaN删掉，而是选择并写明契约：若语言允许NaN，就保留乘法；若某专用内核只接受有限输入，则包装层检查并明确拒绝。编译优化依赖前置条件，测试应验证这些条件实际得到执行。

~~~python
def wrong_zero_rewrite(x):
    return np.zeros_like(x)

fault_x = np.array([3, np.nan], dtype=np.float32)
with np.errstate(invalid="ignore"):
    reference_zero = fault_x * np.float32(0)
wrong_zero = wrong_zero_rewrite(fault_x)
assert np.isnan(reference_zero[1]) and wrong_zero[1] == 0

def correct_zero_product(x):
    return np.multiply(x, np.float32(0))

with np.errstate(invalid="ignore"):
    fixed_zero = correct_zero_product(fault_x)
np.testing.assert_array_equal(fixed_zero, reference_zero)
~~~

第二个故障是把输入数据的计算结果当成编译产物缓存。第一次x=[1,2,3]得到[9,25,49]，第二次x=[4,5,6]仍返回旧数组。源码、shape与dtype都相同，但普通输入值必须每次读取。修复是缓存程序、不是缓存无条件复用的输出；若真的做结果缓存，就要明确把输入内容与可变性纳入另一套语义，不能混同编译缓存。

~~~python
class WrongResultCache:
    def __init__(self):
        self.cached = None

    def run(self, env):
        if self.cached is None:
            self.cached = interpret(tree, env)
        return self.cached

wrong_cache = WrongResultCache()
wrong_cache.run(inputs)
stale = wrong_cache.run(changed_values)
assert np.array_equal(stale, expected)
assert not np.array_equal(stale, interpret(tree, changed_values))
fixed_jit = TinyJIT()
fixed_jit.run(source, inputs)
np.testing.assert_array_equal(
    fixed_jit.run(source, changed_values),
    interpret(tree, changed_values),
)
assert fixed_jit.compilations == 1
~~~

第三个故障是把元素偏移直接当成字节偏移。假设float32数组地址1000，索引3应访问1012；错误地址1003不仅指向错误数据，还可能违反访问对齐。调试时打印源码中的i=3并不能发现这个错误，因为元素索引本身正确，失效发生在降低成字节地址的阶段。用带单位的中间记录可以明确指出哪一步丢失了乘4。

~~~python
def wrong_address(base_bytes, element_index):
    return base_bytes + element_index

def element_address(base_bytes, element_index, itemsize):
    if element_index < 0 or itemsize <= 0:
        raise ValueError("此例只支持非负索引与正元素宽度")
    return base_bytes + element_index * itemsize

assert wrong_address(1000, 3) == 1003
assert element_address(1000, 3, 4) == 1012
assert element_address(1000, 3, 2) == 1006
~~~

真实系统的失败还需要分层收集证据。下表不是让你按错误文本机械分类，而是指引最先检查的对象。同一个最终异常可能包裹更早的内部原因，因此保留完整日志与最小输入，并从最早失败阶段向下追。不要在缺少证据时同时升级驱动、框架和编译器，否则即使问题消失，也无法知道哪一项相关。

| 现象 | 优先检查的层 | 最小证据 |
|---|---|---|
| import失败 | 依赖与环境 | Python解释器、包版本、模块路径 |
| 不支持的语法节点 | 前端语言边界 | 最小表达式与AST节点 |
| 类型或形状推断失败 | 输入合同 | dtype、shape、stride与操作 |
| IR验证失败 | 中间变换 | 变换前后IR与失败条件 |
| 目标指令不支持 | 后端目标选择 | 目标架构、工具版本、指令需求 |
| 未定义符号 | 主机或设备链接 | 声明、定义、参与链接的文件 |
| 装载失败 | 镜像与驱动兼容 | cubin/PTX目标、驱动、设备 |
| 启动参数错误 | 运行提交 | grid、block、共享内存与参数 |
| 数值错误 | 程序与数值语义 | 最小反例、参考结果、误差位置 |
| 首次慢或反复变慢 | 编译缓存与初始化 | 缓存键、编译次数、分段计时 |

最小复现应保留触发问题的特性，而不是任意删代码。若错误只发生在非连续输入，改成连续小数组可能“修好”问题却失去线索；若只发生在尾块，长度改成BLOCK倍数也会掩盖错误。缩小输入时逐步保持dtype、stride、边界、目标与编译选项，记录哪个改变使故障消失，才能把反例变成定位工具。

## 第十四课：十二道递进练习与完整参考答案

### 1. 手画语法树并预测值

题目：对x=2、a=3、b=4，比较x+a*b与(x+a)*b，写出根节点和结果。答案：前者根是add，右子树mul(a,b)，结果14；后者根是mul，左子树add(x,a)，结果20。完整验证如下。若得到相同值，先检查解析结构，不要先怀疑浮点误差，因为这里所有整数都可精确表示。

~~~python
scalar_env = {name: np.array(value, dtype=np.float32)
              for name, value in {"x": 2, "a": 3, "b": 4}.items()}
first = parse_expression("x+a*b")
second = parse_expression("(x+a)*b")
assert first.op == "add" and second.op == "mul"
assert interpret(first, scalar_env).item() == 14
assert interpret(second, scalar_env).item() == 20
~~~

### 2. 加入编译入口资源限制

题目：只允许长度不超过128字符、树深度不超过16的表达式，并验证拒绝路径。答案：先限制字符串长度，再解析，再用显式栈检查深度。长度检查在解析前执行，避免把无限输入交给解析器；深度检查保护后续递归处理。这个教学范围不是对任意不可信输入的完整沙箱保证，但能表达可测试的资源合同。

~~~python
def bounded_parse(source, max_chars=128, max_depth=16):
    if len(source) > max_chars:
        raise ValueError("表达式过长")
    root = parse_expression(source)
    pending = [(root, 1)]
    while pending:
        node, depth = pending.pop()
        if depth > max_depth:
            raise ValueError("表达式过深")
        pending.extend((arg, depth + 1) for arg in node.args)
    return root

assert bounded_parse("x*(2+3)").op == "mul"
for text_value in ("x" * 129, "x+" * 20 + "x"):
    try:
        bounded_parse(text_value)
    except ValueError:
        pass
    else:
        raise AssertionError("资源限制失效")
~~~

### 3. 证明常量折叠保持本例结果

题目：将(x*(2+3))+(4*5)折叠，列出仍需运行时执行的操作。答案是x*5+20，仍有一次乘法与一次加法。不能把x也折叠，因为它属于本次调用输入。下面既检查结构中的算术数，也检查实际数组结果，防止只看打印文本就误判优化已经发生。

~~~python
fold_task = parse_expression("(x*(2+3))+(4*5)")
fold_result = fold_constants(fold_task)
fold_program = lower_to_ssa(fold_result)
arithmetic = [i for i in fold_program.code if i.op in ("add", "sub", "mul", "neg")]
assert len(arithmetic) == 2
np.testing.assert_array_equal(
    execute_ssa(fold_program, inputs),
    np.array([25, 30, 35], dtype=np.float32),
)
~~~

### 4. 检查公共子表达式确实合并

题目：编译(x*a+b)+(x*a+b)，计算优化后算术指令数。答案为三条：一次乘法、一次内层加法、一次最终加法。两个相同子表达式共用值，最终加法仍需要执行。即便最终加法可被进一步改为乘二，这个变换也不属于当前优化器，不能在答案里假装已经实现。

~~~python
common = lower_to_ssa(parse_expression("(x*a+b)+(x*a+b)"))
assert sum(i.op in ("add", "mul") for i in common.code) == 3
assert common.code[-1].inputs[0] == common.code[-1].inputs[1]
np.testing.assert_array_equal(
    execute_ssa(common, inputs), np.array([6, 10, 14], dtype=np.float32)
)
~~~

### 5. 删除两条无用指令

题目：在现有program后添加常量8和9，返回值仍为原结果，执行DCE。答案是恢复原来六条指令，返回编号保持对应关系。若返回值改成新增常量9，旧平方链将全部变成无用代码，只保留常量9。两种情况分别验证从返回值反向可达的规则。

~~~python
extra_code = program.code + (
    Instruction("const", (), constant(8).payload),
    Instruction("const", (), constant(9).payload),
)
assert eliminate_dead_code(Program(extra_code, program.result)) == program
only_nine = eliminate_dead_code(Program(extra_code, len(extra_code) - 1))
assert len(only_nine.code) == 1
assert execute_ssa(only_nine, inputs).item() == 9
~~~

### 6. 解释活跃峰值为何不是精确寄存器数

题目：本章区间模型得到峰值3，能否声称GPU每线程只需要3个寄存器？答案不能。模型按数组逻辑值计数，没有展开每个值的元素数量，没有地址和控制临时量，也没有目标指令约束。一个逻辑数组可能对应许多寄存器，某些指令又允许输入输出重用。这个结果只说明教学SSA图的同时存活值数量，真实资源要看目标编译产物。

### 7. 分析浮点变换的合法条件

题目：将(x+a)+b改成x+(a+b)，应如何验收？答案先明确是否允许重新结合、是否只要求容差、输入是否含非有限值，再用大数抵消、小量累加、正负零和随机有限输入测试。本章严格解释器不做此变换。下面给出一个实际反例，说明仅用小整数测三次通过并不足以证明通用等价。

~~~python
round_env = {
    "x": np.array(100000000, dtype=np.float32),
    "a": np.array(-100000000, dtype=np.float32),
    "b": np.array(1, dtype=np.float32),
}
left_round = interpret(parse_expression("(x+a)+b"), round_env)
right_round = interpret(parse_expression("x+(a+b)"), round_env)
assert left_round.item() == 1 and right_round.item() == 0
~~~

### 8. 区分四种编译产物

题目：.ii、.s、.o、可执行程序分别能证明什么？答案：.ii证明预处理到该阶段完成；.s证明已生成对应主机目标汇编；.o证明已生成目标文件；可执行程序证明链接完成。只有实际启动并检查输出，才增加运行正确性的证据。四者都不能证明CUDA路径已编译，因为本章那组g++命令处理的是主机C++。编译成功与运行成功必须分别记录。

### 9. 手算一次线程到地址的降低

题目：blockIdx=3、blockDim=8、threadIdx=5，float32基地址8192，n=30，求索引、偏移、有效性。答案：索引29，偏移116字节，地址8308，有效。若threadIdx变6，索引30越界，不能读取。若数据改成16位元素，偏移应为58字节；索引不变，元素宽度改变地址。这个题同时检查覆盖、mask和类型宽度。

### 10. 为动态长度设计guard

题目：某版本分配的逻辑块宽为128，只支持一维连续float32输入和长度1到128，写完整guard。答案如下。这里检查byte stride等于itemsize，因此反向视图被拒绝；空数组被拒绝是本题合同，若以后支持空输入，要修改包装返回路径与测试。guard的作用是判断是否能用这个版本，不是自动修复输入。

~~~python
def accepts_block128(value):
    array = np.asarray(value)
    return (
        array.dtype == np.float32
        and array.ndim == 1
        and 1 <= array.size <= 128
        and array.strides == (array.itemsize,)
    )

assert accepts_block128(np.ones(17, dtype=np.float32))
assert not accepts_block128(np.ones(129, dtype=np.float32))
assert not accepts_block128(np.ones(17, dtype=np.float64))
assert not accepts_block128(np.ones(17, dtype=np.float32)[::-1])
assert not accepts_block128(np.empty(0, dtype=np.float32))
~~~

### 11. 解释缓存键的反例

题目：只用shape作为键会漏掉什么？答案包括源码、dtype、stride、目标、编译器与配置。相同shape的float16和float32元素宽度不同，相同shape的连续与转置视图地址不同，目标专用指令也可能不同。下面直接证明本章键区分配置与步长，而普通输入数值变化保持键相同。这样验证的是缓存策略，不是依靠时间差猜测命中。

~~~python
base_key = compilation_key(source, inputs, "numpy-cpu", {})
assert base_key == compilation_key(source, changed_values, "numpy-cpu", {})
assert base_key != compilation_key(source, reversed_view, "numpy-cpu", {})
assert base_key != compilation_key(source, inputs, "other-target", {})
assert base_key != compilation_key(source, inputs, "numpy-cpu", {"tile": 64})
~~~

### 12. 构造完整差分测试矩阵

题目：为四个表达式、四种长度、连续与负步长输入比较解释器与优化SSA，至少覆盖空长度与非整齐长度。完整答案如下，总计32个案例。随机种子固定使失败可复现，值域较小避免本题被溢出掩盖。逐元素结果相等后再比较位模式，检查包括符号零在内的表示；测试范围明确，不把32例当作所有程序的形式化证明。

~~~python
generator = np.random.default_rng(20261101)
expressions = (
    "(x*a+b)*(x*a+b)",
    "x*(2+3)+4*5",
    "(x-a)*(x+a)",
    "-x+b",
)
case_count = 0
for expression in expressions:
    root = parse_expression(expression)
    optimized_program = lower_to_ssa(fold_constants(root))
    for length in (0, 1, 7, 17):
        raw = generator.uniform(-3, 3, size=length).astype(np.float32)
        for x_view in (raw, raw[::-1]):
            environment = {
                "x": x_view,
                "a": np.array(0.75, dtype=np.float32),
                "b": np.array(-0.25, dtype=np.float32),
            }
            reference = interpret(root, environment)
            actual = execute_ssa(optimized_program, environment)
            np.testing.assert_array_equal(actual, reference)
            np.testing.assert_array_equal(
                actual.view(np.uint32), reference.view(np.uint32)
            )
            case_count += 1
assert case_count == 32
print("differential cases:", case_count)
~~~

## 第十五课：综合项目——一份能解释每次变换的编译作品

第一阶段把本章Python代码整理成一个小模块，分成语法表示、解释器、优化、SSA执行、缓存与测试六个清晰部分。保留原始输入表达式、AST节点数、SSA指令数、常量折叠前后表示和缓存编译次数。程序每次执行应输出结果与这些可解释信息，而不只打印一个“编译成功”。你需要能说明为什么某次减少两条算术、为什么换输入值没有重新编译、为什么负步长触发新版本。

第二阶段扩展一个操作，例如逐元素除法。先定义除零、非有限值和float32舍入语义，再修改解析白名单、解释器、形状推断与测试，最后考虑常量折叠。不要直接加一行运算符映射就宣布完成。若选择不支持除零，要证明运行输入和编译期常量都在相应阶段被拒绝；若选择遵循NumPy浮点行为，要保留警告与结果的明确策略。

第三阶段完成C++编译产物实验，保存源文件、完整命令、编译器版本、O0与O2产物摘要、程序输出和一处能解释的数据流。报告中可以说“本机优化构建采用了某种形式”，不能说“所有编译器一定生成这条指令”。如果main被常量折叠，解释为什么，再引入运行时输入观察变化。实验价值在于把可观察证据和源语言语义连接起来，而不是截一张汇编截图。

第四阶段在获分配的单个GPU上完成CUDA产物与运行分离验证。先生成PTX和cubin，记录目标与工具版本；再用K03的完整主机包装运行小规模add，检查数值与尾部。随后读取实际PTX与SASS，追一个有效线程和一个尾部线程。这个任务不需要长期占用H100，也不需要八卡；只有目标专用研究才选择对应设备，普通编译链学习可以先在个人环境进行。

第五阶段选择K09一个Triton算子，记录输入合同、编译期参数、实际版本数与冷暖调用。只比较三个有限shape和两个配置，解释每次缓存变化的原因。若要查看内部IR，先检查安装版本提供的官方或源码接口，记录实际使用方式，不把网上旧代码里的私有属性当永久API。任何未能获取的阶段写明未观察，不凭相邻阶段推测它必然使用了某条优化。

最终验收要求你能从一个错误结果反向提出有顺序的假设：输入合同是否成立，AST是否表达原意，变换是否保持数值与效果，地址降低是否保留单位，目标产物是否兼容，运行参数是否正确。你还应能说明同一个“慢”发生在编译、装载、提交还是设备执行。掌握这条证据链后，再进入LLVM pass、MLIR dialect、Triton后端与目标调优，才有扎实的代码阅读支点。

一手依据：[NVCC编译驱动文档](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)、[LLVM语言参考](https://llvm.org/docs/LangRef.html)、[CUDA二进制工具](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html)、[PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)。核查日期为2026-11-01；工具内部步骤与选项以实际安装版本为准。本章小语言及优化器是独立教学实现，不宣称替代这些真实编译器。
