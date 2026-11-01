# F13 Docker 从镜像到服务：知道你的程序究竟住在哪里

> 先修 F04、F05、F10。离线模型实验：`python -m labs.foundations.container_model`。真实Docker步骤是明确的手动拓展，不会在自动测试中拉镜像或启动容器。

## 1. 把四个常混淆的名字拆开

Docker是一套构建与运行容器的工具体系；镜像是一份只读模板；容器是从镜像创建出来的运行实例，通常有自己的可写层、配置和进程；Compose是描述并管理一组服务配置的工具。它们不是同义词。

你可以从同一个镜像启动两个容器，一个监听8001，一个监听8002，它们各自有进程状态和可写层。停止其中一个不会自动停止另一个；删除容器不等于删除原镜像；拉取一个新镜像不等于正在运行的旧容器自动升级。

Docker通常采用客户端与daemon分离的架构。终端中的docker命令是客户端，它向daemon请求创建、运行、查看对象。当前连接哪个daemon很重要：本地Docker Desktop、远端Linux Docker或不同context可能管理不同资源。看到一组容器，不代表你正看公司所有服务器。

## 2. 镜像层与容器可写层

镜像由层和配置等内容组成，层可复用。新镜像和旧镜像可能共享大量层，因此拉两个版本不一定下载两份完全独立的数据。tag是可移动的名字，例如某个nightly标签今天和明天可能指向不同内容；digest标识具体内容，适合复现时固定。

容器运行时对文件的修改通常落在可写层。删除这个容器后，那些未放到持久存储的修改就不再属于新容器。只停止再启动同一个容器，其可写层通常仍在。把stop和rm区分开，才能预测数据是否保留。

离线模型使用image字典、writable字典和volume字典。第一个容器把/app/version改成temporary-v2，把/data/progress写成lesson13；第二个新容器复用原镜像和同一个volume，因此版本回到v1，而progress仍是lesson13。

它是生命周期概念模型，不模拟真实overlay filesystem、权限、copy-on-write粒度和mount优先级。不要把字典查找的性能推断成真实Docker读写速度。

## 3. Volume、bind mount和构建上下文

volume由Docker管理，适合持久数据。bind mount把宿主机某个路径映射到容器路径，适合开发源码或明确的数据目录。容器里的路径可能与宿主机完全不同，所以“/data”必须同时说明位于哪一侧。

bind mount可以只读。读取模型权重或输入数据时，若程序不需要修改它们，可以使用只读挂载，减少误改。运行用户的UID/GID与目录权限也会影响读写，不能把所有permission denied都归咎于Docker。

构建上下文是docker build可发送给构建器的文件范围。若在含巨大权重和密钥的目录直接build，又没有正确.dockerignore，可能把不必要内容送入构建上下文。`.gitignore`与`.dockerignore`解决不同工具的输入筛选，不会自动替代彼此。

## 4. 用课程日志工具构建一个小镜像

本卷提供 `labs/foundations/Dockerfile`，从仓库根目录构建：

```bash
docker build -f labs/foundations/Dockerfile -t llm-course-foundations:local .
docker run --rm llm-course-foundations:local
```

这是明确需要已安装Docker、可获得基础镜像的操作；教材自动验证没有执行它。镜像只复制labs包和foundations子包，不复制整个仓库，不安装GPU依赖，不需要GPU。默认运行日志分析demo，完成后退出；`--rm`只删除这次自动创建的容器，不删除镜像。

Dockerfile中使用Python 3.12系列基础标签以表达兼容范围；正式复现实验应记录构建时解析出的digest，不能把可变标签视为永远相同。课程没有编造一个未核查的digest。

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY labs/__init__.py /app/labs/__init__.py
COPY labs/foundations /app/labs/foundations
USER 10001:10001
ENTRYPOINT ["python", "-m", "labs.foundations.log_project"]
```

JSON数组形式ENTRYPOINT直接描述可执行文件和参数，避免额外Shell层的引号与信号语义。这个小镜像不需要写/app，非root用户足够。若增加输出挂载，要确保挂载目标对该用户可写，而不是直接改回root掩盖权限设计。

## 5. 端口、监听地址和容器网络

端口发布`-p 127.0.0.1:8765:8765`把宿主机回环8765映射到容器8765。冒号左右分别是宿主机和容器视角。容器内服务若只绑定容器的127.0.0.1，普通bridge映射不一定能到达它；生产容器常在容器内绑定0.0.0.0，再由宿主机发布地址控制外部可达性。

本卷任务服务为了本地学习固定绑定127.0.0.1，所以Docker演练选择不监听端口的日志工具。不要为了让示例“看起来完整”把一个未鉴权教学服务默认暴露到所有接口。以后服务卷会明确设计容器内监听与宿主机发布的组合。

`docker network ls`列网络对象；`docker compose ls`列Compose项目；`docker ps`列容器。它们回答不同问题，没有谁是“更完整的全部Docker列表”。

## 6. 容器不是完整虚拟机，也不是万能保险箱

Linux容器共享宿主内核，同时通过命名空间、cgroups和权限等机制组织隔离与资源限制。镜像里的用户空间与宿主内核共同决定行为。容器可以限制CPU和内存，但GPU可见性、显存占用和公司调度规则需要另外处理。

避免给课程实验使用privileged、挂载Docker socket、挂载整个宿主根目录等高权限设置。它们可能让容器获得远超学习所需的控制。不是因为课程代码“应该没问题”就可以忽略权限边界。

Docker组访问在许多部署中相当于很高的主机控制能力，因此不能把加入docker组当作无风险的普通文件权限调整。公司服务器应沿用管理员认可的使用方式；若需要sudo docker，脚本中每条docker调用的权限问题要有明确方案。

## 7. 日志、停止与资源清理

`docker logs 容器名`读取容器标准输出/错误的日志；`docker stop 容器名`请求停止；`docker rm 容器名`删除已停止容器；`docker image ls`查看镜像。清理只指向你明确创建的对象，课程不使用全局prune或批量删除他人资源。

前台启动容器时，当前终端会附着其输出；`-d`表示后台运行并返回容器ID。后台不是“不会占资源”，模型服务即使没有请求也可能持续占显存。用完GPU实验后应停止自己的实例，验证资源释放。

容器重建后依赖不见了，是因为你只在旧容器可写层临时安装。可复现依赖应写进Dockerfile或明确环境文件；容器运行中手动修复可以用于定位问题，但要把最终修复回写到可重建配置。

## 8. 八个练习与参考答案

**F13-1 生命周期预测。** 修改容器内文件，stop/start同一容器与rm后创建新容器分别怎样？答案：前者通常保留可写层，后者从镜像重新开始；volume可独立保留。

**F13-2 tag与digest。** 为什么复现nightly时应记录digest？答案：tag可移动，digest固定内容；本地名字与内容标识不是一一永恒对应。

**F13-3 只读挂载。** 设计一个只读取输入日志的容器调用，输入bind mount为只读。验收：不挂载整个用户目录，不把密钥放进镜像。

**F13-4 构建边界。** 解释为什么只COPY课程需要的包比COPY整个仓库更清楚。答案：缩小输入范围与意外带入内容，但.dockerignore仍用于减少构建上下文传输。

**F13-5 端口视角。** 宿主映射8766:8765时，客户端访问哪个？答案：宿主8766；容器服务监听8765。再解释容器内localhost不是宿主localhost。

**F13-6 故障分类。** 运行失败分别出现镜像不存在、权限拒绝、程序异常。验收：定位拉取/daemon访问/应用执行层，不一律重新安装Docker。

**F13-7 可复现报告。** 构建并运行课程镜像，记录镜像ID、基础digest、命令和输出。验收：不把“构建成功”宣称为所有宿主平台都已验证。

**F13-8 Compose阅读。** 打开提供的compose.yaml，解释service、image、build、command与profiles。答案：profile使练习服务显式选择；up启动的是实例，不是“安装镜像到主机系统”。

## 掌握门槛

你能预测镜像、容器层与volume的生命周期，能构建一个最小非GPU工具镜像，能明确解释端口和挂载两侧路径，并能只清理自己的资源。

原始资料：[Docker overview](https://docs.docker.com/get-started/docker-overview/)，入口核查于2026-11-01。
