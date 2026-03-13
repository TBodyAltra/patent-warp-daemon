# 专利申请技术交底书

---

## IP审核人填写

- **案号**：_________________
- **交底书提交日**：_________________
- **新申请委托日**：_________________

### 审核意见

- ☐ 发明； ☐ 实用新型； ☐ 外观设计； ☐ 作为技术秘密保护； ☐ 不申请。

### 申请人

- ☑ 成都海光集成电路设计有限公司； ☐ 成都海光微电子技术有限公司； ☐ 海光信息技术有限公司。

---

## 技术审核人填写

- **技术审核人**：_________________
- **部门**：_________________
- **审核意见**：_________________

---

## 发明人填写

### 技术问题联系人

- **姓名**：_________________
- **电话**：_________________
- **E-mail**：_________________

### 第一发明人

- **第一发明人姓名**：_________________
- **第一发明人身份证号**：_________________

---

## 技术方案名称

**基于运行时动态调度的 Warp 分工方法及系统**

### 涉及的项目

_________________

### 关键词

- **中文**：Warp 调度、GPU 编译器、Warp-Specialize、动态调度、Kernel 优化、DCU、Triton、硬件资源利用、线程束特化
- **英文**：Warp Scheduling, GPU Compiler, Warp-Specialize, Dynamic Scheduling, Kernel Optimization, DCU, Triton, Hardware Resource Utilization, Thread Warp Specialization

---

## 技术方案实施情况

- ☑ 未列入公司产品计划； ☐ 实施阶段，将在公司新产品中实施，拟应用产品为：_________________

---

## 缩略语和关键术语定义

| 术语 | 定义 |
|------|------|
| Warp | GPU 中的基本执行单元，AMD GPU 中为一个 Wave（64 线程），NVIDIA GPU 中为 32 线程的线程束 |
| Warp-Specialize | Warp 特化技术，将一个 Warp 的线程划分为不同角色（Producer/Consumer），分别执行内存访问和计算 |
| Producer | Warp-Specialize 中负责数据加载和内存操作的线程角色 |
| Consumer | Warp-Specialize 中负责计算操作的线程角色 |
| Warpgroup | AMD GPU 中的一组 Warps，用于协同执行任务 |
| VGPR | 向量通用寄存器（Vector General-Purpose Register），GPU 中用于存储线程执行数据的寄存器 |
| SGPR | 标量通用寄存器（Scalar General-Purpose Register），GPU 中用于存储线程间共享数据的寄存器 |
| Kernel | GPU 核函数，在 GPU 上执行的并行计算函数 |
| Partition | 分区，编译时将 Kernel 按硬件单元使用类型划分成的不同部分 |
| Task | 任务，Partition 按数据维度切分后的最小执行单元 |
| Daemon Warp | 调度器 Warp，运行时负责监控硬件状态并调度任务执行的特殊 Warp |
| IR | 中间表示（Intermediate Representation），编译器中用于表示源代码的中间格式 |
| SIMD | 单指令多数据（Single Instruction Multiple Data），GPU 的并行执行模式 |
| MFMA | 矩阵融合乘加（Matrix Fused Multiply-Add），AMD GPU 的矩阵运算指令 |

---

## 背景技术

### 1.1 技术领域

本发明属于 GPU 编译器优化技术领域，具体涉及一种面向国产 GPU（如海光 DCU）的 Warp 级动态调度方法及系统，特别涉及基于编译时分区和运行时动态任务调度的 Warp 分工优化技术。

### 1.2 相关技术背景

随着人工智能技术的快速发展，GPU 作为最重要的并行计算硬件，在深度学习、科学计算等领域发挥着关键作用。为了充分发挥 GPU 的计算性能，编译器优化技术一直是研究的热点方向。

在 GPU 编程中，Warp-Specialize（线程束特化）是一种重要的优化技术。其核心思想是将一个 Warp 组的线程划分为不同的角色：一部分线程作为"生产者（Producer）"负责数据加载和内存访问，另一部分线程作为"消费者（Consumer）"负责计算操作。通过让 Producer 和 Consumer 并行工作，可以实现计算与内存访问的重叠，减少硬件空闲时间，提高整体吞吐量。

当前，Warp-Specialize 技术已被广泛应用于高性能 GPU Kernel 的开发中。例如：

1. **NVIDIA FlashAttention 系列**：使用 Producer 线程预取数据，Consumer 线程执行注意力计算
2. **AMD HIPKittens**：提出了 8-wave ping-pong 调度模式
3. **Triton 编译器**：提供了 Warp-Specialize 的语言层面的支持

### 1.3 现有技术分析

#### 1.3.1 现有技术一：静态 Warp-Specialize

**技术方案**：在编译阶段静态划分 Warp 的角色，运行时 Producer 和 Consumer 的分工固定不变。

**代表实现**：Triton 语言的 `tt.create_group` 原语

**缺点**：
- 在编译时确定角色，无法适应运行时变化
- 当 Producer 和 Consumer 工作量差异较大时，一方会空闲等待
- 无法根据硬件资源占用情况动态调整任务分配

#### 1.3.2 现有技术二：HipKittens 8-wave Ping-Pong

**技术方案**：AMD HipKittens 框架提出的 8-wave ping-pong 调度模式，让 Warp 在不同阶段扮演不同角色。

**论文**：HipKittens: Fast and Furious AMD Kernels (arXiv:2511.08083)

**缺点**：
- 主要解决 AMD CDNA 架构的静态寄存器分配问题
- 角色交换是按固定阶段进行的，无法响应运行时硬件状态变化
- 仍然在编译时确定整体策略

#### 1.3.3 现有技术三：动态指令重排

**技术方案**：通过运行时动态调整指令顺序来优化执行效率。

**代表实现**：NVIDIA CUDA 的动态并行特性

**缺点**：
- 依赖硬件特定的动态调度能力
- 开销较大，不适合所有场景
- 可移植性差

#### 1.3.4 现有技术四：编译器指导的 Warp 分工

**技术方案**：通过编译器分析代码特征，指导 Warp 分工。

**代表实现**：各类研究原型系统

**缺点**：
- 仍然是静态分析，无法获取运行时信息
- 对复杂数据依赖的处理能力有限

### 1.4 现有技术总结

| 技术方案 | 分工时机 | 负载均衡 | 资源利用 | 适应性 |
|---------|---------|---------|---------|-------|
| 静态 Warp-Specialize | 编译时 | 差 | 中 | 差 |
| HipKittens Ping-Pong | 编译时+固定阶段 | 中 | 高 | 中 |
| 动态指令重排 | 运行时 | 好 | 高 | 好（但开销大） |
| 编译器指导 | 编译时 | 差 | 中 | 差 |
| **本发明** | **编译时分区+运行时动态** | **好** | **高** | **好** |

**与现有技术的区别**：本发明首次提出了"编译时分区+运行时动态调度"的架构，通过引入 Daemon Warp 机制，实现了真正响应运行时硬件状态变化的动态 Warp 分工，在保证低开销的前提下实现了最优的资源利用率。

---

## 发明内容

### 1. 发明目的

本发明的目的是提供一种基于运行时动态调度的 Warp 分工方法及系统，以解决现有技术中静态划分导致负载不均衡、资源利用率低的问题，实现 GPU Kernel 的自适应优化执行。

### 2. 技术方案概述

一种基于运行时动态调度的 Warp 分工方法及系统，其核心思想是：

1. **编译阶段**：将 Kernel 按照使用的硬件单元划分为不同的分区（Partition），每个分区再按照数据维度切分为小的任务（Task），预计算每个 Task 需要的输入（内存地址、参数等），存储于共享内存的结构体中

2. **运行阶段**：由一个特殊的 Warp 担任调度器（Daemon Warp），持续监控硬件资源占用情况，当有 Warp 空闲时，根据硬件状态选择合适的 Partition，根据数据依赖关系选择满足条件的 Task，分发给空闲的 Warp 执行

3. **重复执行**：重复上述过程直到所有 Task 执行完毕

### 3. 详细技术方案

#### 3.1 编译阶段

##### 3.1.1 Kernel 分区划分

在编译阶段，分析 Kernel 代码的 IR（中间表示），识别不同类型的硬件操作：

- **内存分区（Partition-M）**：主要执行内存访问操作（Load/Store）的代码区域
- **计算分区（Partition-C）**：主要执行计算密集型操作（MFMA、ALU）的代码区域
- **混合分区（Partition-H）**：同时包含内存和计算操作的代码区域

```
┌─────────────────────────────────────────────────────────────┐
│                    原始 Kernel IR                          │
├─────────────────────────────────────────────────────────────┤
│  for (i = 0; i < N; i++) {                               │
│      A[i] = load(input + i);   // → Partition-M          │
│      B[i] = A[i] * A[i];       // → Partition-C          │
│      C[i] = B[i] + bias;       // → Partition-C          │
│      store(output + i, C[i]);   // → Partition-M          │
│  }                                                         │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                    分区后的 IR                              │
├─────────────────────────────────────────────────────────────┤
│  Partition-M:  [load] [store]                              │
│  Partition-C:  [mul] [add]                                 │
│  Partition-H:  [load] [mul] [add] [store]                 │
└─────────────────────────────────────────────────────────────┘
```

##### 3.1.2 Task 切分与输入预计算

每个 Partition 按照数据维度进一步切分为多个 Task，并预计算每个 Task 的输入：

```c
// 编译生成的 Task 描述符结构
struct TaskDescriptor {
    uint32_t task_id;           // Task 唯一标识
    uint32_t partition_id;       // 所属分区类型
    uint32_t input_ptr;          // 输入数据在共享内存的偏移量
    uint32_t output_ptr;         // 输出数据在共享内存的偏移量
    uint32_t param_offset;       // 参数在共享内存的位置
    uint32_t depend_mask;        // 依赖 Task 掩码（哪些 Task 必须先完成）
    uint32_t ready_mask;         // 前置依赖完成状态（运行时更新）
    void (*func_ptr)(void*);   // Task 执行函数指针
    uint32_t vgpr_needed;       // 需要分配的 VGPR 数量
    uint32_t shared_mem_size;    // 需要分配的共享内存大小
};
```

**输入预计算示例**：

```
原始代码:
  for (int i = tile_start; i < tile_end; i++) {
      A[i] = load(base + i * stride);
  }

编译时预计算:
  TaskDescriptor:
    .input_ptr = shared_mem_base + tile_id * tile_size
    .param_offset = stride  // 预计算好的参数
    .depend_mask = 0  // 无依赖
```

##### 3.1.3 调度代码生成

编译阶段生成两部分代码：

1. **Task 执行代码**：每个 Task 的具体实现编译成的机器码
2. **Daemon 调度代码**：运行时 Daemon Warp 执行的调度逻辑

#### 3.2 运行阶段

##### 3.2.1 初始化

Kernel 启动时，将 Task 描述符数组加载到共享内存：

```
┌─────────────────────────────────────────────────────────────┐
│                    共享内存布局                            │
├─────────────────────────────────────────────────────────────┤
│  [Task 0 描述符] [Task 1 描述符] [Task 2 描述符] ...    │
├─────────────────────────────────────────────────────────────┤
│  [Warp 空闲状态表] - 每个 Warp 的状态（空闲/忙碌）        │
├─────────────────────────────────────────────────────────────┤
│  [硬件状态监控区] - 内存/计算单元利用率                    │
├─────────────────────────────────────────────────────────────┤
│  [调度队列] - 按优先级排序的就绪 Task 列表                │
└─────────────────────────────────────────────────────────────┘
```

##### 3.2.2 Daemon Warp 调度主循环

```c
// Daemon Warp 伪代码
__device__ void daemon_scheduler() {
    while (!all_tasks_completed()) {
        
        // 1. 监控硬件资源占用
        HardwareStatus hw_status = read_hardware_counters();
        
        // 2. 查找空闲 Warp
        int idle_warp = find_idle_warp();
        if (idle_warp == -1) {
            continue;  // 没有空闲 Warp，等待
        }
        
        // 3. 根据硬件状态选择分区类型
        int partition = select_partition(hw_status);
        
        // 4. 在选中分区中找满足依赖的 Task
        Task* task = find_ready_task(partition);
        if (task == nullptr) {
            continue;  // 没有就绪 Task
        }
        
        // 5. 更新 Task 状态
        atomic_and(&task->ready_mask, 0);  // 标记为已分发
        
        // 6. 唤醒空闲 Warp 执行 Task
        wake_warp(idle_warp, task);
    }
}

// 分区选择策略
__device__ int select_partition(HardwareStatus hw) {
    if (hw.memory_util < 0.5 && hw.compute_util > 0.8) {
        return PARTITION_MEMORY;  // 计算单元忙，分配内存任务
    } else if (hw.memory_util > 0.8 && hw.compute_util < 0.5) {
        return PARTITION_COMPUTE;  // 内存带宽忙，分配计算任务
    } else if (hw.memory_util > 0.8 && hw.compute_util > 0.8) {
        return PARTITION_HYBRID;   // 都忙，选择混合任务
    } else {
        return PARTITION_COMPUTE;  // 默认优先计算
    }
}

// 查找满足依赖的 Task
__device__ Task* find_ready_task(int partition_type) {
    for (int i = 0; i < total_tasks; i++) {
        Task* t = &task_array[i];
        
        // 检查分区类型
        if (t->partition_id != partition_type) continue;
        
        // 检查依赖是否满足
        if ((t->depend_mask & completed_mask) == t->depend_mask) {
            return t;  // 依赖已满足
        }
    }
    return nullptr;
}
```

##### 3.2.3 完整执行流程

```
时间 →
──────────────────────────────────────────────────────────────────────→

Warp 0(Daemon):  [监控][调度][监控][调度][监控][调度][监控][完成]

Warp 1:          [Task0][    ][Task3][    ][Task6][    ][    完成]

Warp 2:          [  ][Task1][    ][Task4][    ][Task7][    ][   完成]

Warp 3:          [  ][  ][Task2][    ][Task5][    ][Task8][     完成]

Warp 4:          [  ][  ][  ][Task0][    ][Task3][    ][Task6][完成]

... (持续动态调度)

共享内存状态:
  t0: Task0完成 → 更新completed_mask → Task1,2就绪
  t1: Task1完成 → 更新completed_mask → Task3就绪
  t2: Task2完成 → 更新completed_mask → Task4就绪
  ...

硬件资源利用:
  内存带宽:  ████████████░░░░░░░  ~80%
  计算单元:  ██████████████░░░░░  ~85%
```

#### 3.3 与现有技术的核心区别

| 对比项 | 传统 Warp-Specialize | 本发明方案 |
|--------|---------------------|------------|
| 分工时机 | 编译时静态 | 编译时分区 + 运行时动态 |
| 任务分配 | 固定角色 | Daemon 根据状态动态分配 |
| 负载均衡 | 静态，可能不均衡 | 动态适应，始终均衡 |
| 资源利用 | 部分利用 | 最大化利用 |
| 适应性 | 差 | 好（响应硬件状态） |
| 开销 | 低 | 低（仅一个 Daemon Warp） |

---

## 技术效果

本发明与现有技术相比，具有以下显著的技术效果：

| 序号 | 效果 | 说明 |
|------|------|------|
| 1 | **负载均衡** | 通过运行时动态调度，自动适应不同阶段的负载变化，避免 Warp 空闲 |
| 2 | **资源最大化利用** | Daemon 根据硬件资源占用情况智能分配任务，始终保持高利用率 |
| 3 | **低开销** | 仅使用一个 Warp 作为 Daemon，开销极低 |
| 4 | **高灵活性** | 可处理复杂的条件分支和动态数据依赖 |
| 5 | **易于实现** | 编译器自动完成分区和 Task 切分，无需开发者手动管理 |
| 6 | **可扩展性强** | 可容易地添加新的调度策略 |
| 7 | **适配国产 GPU** | 针对 DCU 特性设计，发挥硬件特有能力（如差异化 VGPR 分配） |

---

## 具体实施方式

### 实施例一：矩阵乘 Kernel

1. 编译阶段分析矩阵乘 Kernel，识别出：
   - Partition-M：加载 A、B 矩阵块
   - Partition-C：MFMA 计算
   - Partition-H：计算 + 存储结果

2. 按数据维度切分 Task：
   - 每个输出块作为一个 Task
   - 预计算每个块在共享内存的位置

3. 运行时 Daemon 调度：
   - 当计算单元忙时，分配更多内存任务
   - 当内存带宽满时，分配更多计算任务

### 实施例二：Attention Kernel

1. 编译阶段识别：
   - QKV 加载 → Partition-M
   - softmax 计算 → Partition-C
   - 上下文存储 → Partition-M

2. 数据依赖关系：
   - QKV 加载完成后才能执行 softmax
   - softmax 完成后才能存储结果

3. 运行时 Daemon 确保依赖满足后调度

### 实施例三：混合负载场景

对于同时包含计算密集型和内存密集型的复杂 Kernel：

1. 编译时识别不同操作类型
2. 运行时 Daemon 监控：
   - 内存利用率低 → 优先分配内存任务
   - 计算利用率低 → 优先分配计算任务
   - 两者都高 → 分配混合任务

---

## 发散思维

1. **多 Daemon 架构**
   - 多个 Daemon Warp 协同调度
   - 适用于大规模并行场景

2. **层级调度**
   - 在 Warp 之上引入线程块级别的调度
   - 实现更粗粒度的负载均衡

3. **学习型调度**
   - 使用历史执行数据训练调度模型
   - 预测最优分配策略

4. **跨 Kernel 调度**
   - 将多个相关 Kernel 联合优化
   - 减少跨 Kernel 的数据移动

5. **异构调度**
   - 结合 CPU、GPU、专用加速器的混合调度
   - 实现系统级最优

---

## 发明人认为本技术方案最有价值的保护点

### 1. 编译时分区 + 运行时动态调度的架构设计

本发明首次提出了将 Kernel 在编译时按硬件单元使用类型划分为不同分区，并在运行时由 Daemon Warp 根据硬件状态动态调度任务的架构。这种"编译时分析 + 运行时决策"的模式，兼顾了编译优化的确定性和运行时自适应的灵活性，在保证低开销的前提下实现了最优的资源利用率。

### 2. Task 输入预计算机制

通过在编译阶段预计算每个 Task 需要的输入（内存地址、参数等），并存储于共享内存的结构体中，运行时 Daemon 可以快速获取任务执行所需的所有信息，大幅降低了调度开销。

### 3. 硬件状态感知的调度策略

Daemon Warp 通过读取硬件性能计数器，实时感知内存带宽和计算单元的利用情况，据此动态选择应该调度的分区类型，实现真正的自适应负载均衡。

### 4. 依赖感知的任务选择

通过 Task 描述符中的依赖掩码（depend_mask）和完成状态（ready_mask），Daemon 可以准确判断哪些 Task 满足执行条件，避免数据竞争和依赖违例。

### 5. 与 DCU 特性的结合

本方案针对国产 DCU GPU 的特性设计，可以充分发挥 DCU 的差异化 VGPR 分配能力，为 Producer 分配少量 VGPR，为 Consumer 分配大量 VGPR，实现资源的最优利用。

---

## 附图及说明

### 图1：传统静态 Warp-Specialize 示意

展示现有技术中 Producer 和 Consumer 角色固定，运行时无法调整导致的空闲问题。

### 图2：编译阶段 Kernel 分区划分

展示如何将原始 Kernel IR 按硬件操作类型划分为不同分区。

### 图3：Task 切分与数据结构

展示 Task 描述符的结构和依赖关系的表示方法。

### 图4：共享内存布局

展示运行时共享内存中 Task 描述符、Warp 状态表、硬件监控区的布局。

### 图5：Daemon Warp 调度流程

展示 Daemon Warp 的完整调度循环：监控硬件 → 查找空闲 Warp → 选择分区 → 选择 Task → 分发执行。

### 图6：执行时间线示意

展示多个 Warp 在 Daemon 调度下动态执行 Task 的时间线。

---

*文档生成时间：2026-03-13*
